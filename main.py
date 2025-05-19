import uuid
import datetime
from typing import List, Dict, Any, Optional
import json
from pydantic import BaseModel, Field
import pytz
import pathlib

import gradio as gr
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent, ToolNode
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, TypedDict

# Import voice utils
from voice_utils import text_to_speech, speech_to_text, save_audio_to_temp_file, cleanup_temp_file

# Import email helper
from email_helper import send_bulk_email

# Import iCloud CalDAV functions
from ical_test import (
    discover_caldav_calendars,
    list_calendars as list_icloud_calendars,
    get_apple_calendar_events,
    add_event_to_calendar,
    update_event_in_calendar,
    delete_event_from_calendar
)

# Load configuration
def load_config():
    config_path = pathlib.Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# Load credentials from config
config = load_config()
OPENAI_API_KEY = config.get('openai', {}).get('api_key')
OPENAI_API_BASE_URL = config.get('openai', {}).get('api_base_url')
CALENDAR_NAME = config.get('calendar', {}).get('calendar_name')


# Data persistence settings
DATA_DIR = pathlib.Path("calendar_data")
USER_DATA_FILE = DATA_DIR / "user_calendars.json"

# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)

# Create memory saver for conversation persistence
memory_saver = MemorySaver()

# Initialize LLM model
llm = init_chat_model(
    model= "gpt-4o-mini-2024-07-18",
    # model="qwen2.5:3b-mini-2024-07-18",
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE_URL,
    temperature=0.0
)


# Utility functions for date and time handling
def get_current_datetime(timezone_str: str = "UTC+8"):
    """Get the current date and time in the specified timezone"""
    try:
        tz = pytz.timezone(timezone_str)
        return datetime.datetime.now(tz)
    except pytz.exceptions.UnknownTimeZoneError:
        # Default to UTC if timezone is unknown
        return datetime.datetime.now(pytz.UTC)


def get_current_date(timezone_str: str = "UTC+8"):
    """Get the current date in the specified timezone"""
    return get_current_datetime(timezone_str).date()


def format_datetime_iso(dt: datetime.datetime):
    """Format a datetime object as ISO format string"""
    return dt.isoformat()


def parse_datetime_from_iso(iso_str: str):
    """Parse an ISO format string into a datetime object"""
    return datetime.datetime.fromisoformat(iso_str)


# Simulated calendar data storage
class Event:
    def __init__(
            self,
            id: str,
            title: str,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
            calendar_type: str = "personal",
            location: str = "",
            description: str = "",
            attendees: List[str] = None,
            reminders: List[Dict[str, Any]] = None,
            recurring: Dict[str, Any] = None
    ):
        self.id = id
        self.title = title
        self.start_time = start_time
        self.end_time = end_time
        self.calendar_type = calendar_type
        self.location = location
        self.description = description
        self.attendees = attendees or []
        self.reminders = reminders or []
        self.recurring = recurring or {}

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "calendar_type": self.calendar_type,
            "location": self.location,
            "description": self.description,
            "attendees": self.attendees,
            "reminders": self.reminders,
            "recurring": self.recurring
        }


class CalendarManager:
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.events = {}
        self.calendars = ["personal", "work"]
        self.icloud_calendar_name = CALENDAR_NAME  # 默认使用的 iCloud 日历名称
        
        # 初始化 iCloud CalDAV 连接
        self.caldav_url = discover_caldav_calendars()
        if not self.caldav_url:
            print("Warning: Failed to connect to iCloud CalDAV")
        
        # 加载本地数据
        self.load_data()

    def create_event(
            self,
            title: str,
            start_time: datetime.datetime,
            end_time: datetime.datetime,
            calendar_type: str = "personal",
            location: str = "",
            description: str = "",
            attendees: List[str] = None,
            reminders: List[Dict[str, Any]] = None,
            recurring: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create a new calendar event in both iCloud and local storage"""
        # 创建本地事件
        event_id = str(uuid.uuid4())
        event = Event(
            id=event_id,
            title=title,
            start_time=start_time,
            end_time=end_time,
            calendar_type=calendar_type,
            location=location,
            description=description,
            attendees=attendees,
            reminders=reminders,
            recurring=recurring
        )
        self.events[event_id] = event
        
        # 同步到 iCloud
        try:
            success = add_event_to_calendar(
                self.icloud_calendar_name,
                title,
                start_time,
                end_time
            )
            if not success:
                print(f"Warning: Failed to sync event to iCloud: {title}")
        except Exception as e:
            print(f"Error syncing to iCloud: {str(e)}")
        
        # 保存本地数据
        self.save_data()
        return event.to_dict()

    def update_event(
            self,
            event_id: str,
            title: Optional[str] = None,
            start_time: Optional[datetime.datetime] = None,
            end_time: Optional[datetime.datetime] = None,
            calendar_type: Optional[str] = None,
            location: Optional[str] = None,
            description: Optional[str] = None,
            attendees: Optional[List[str]] = None,
            reminders: Optional[List[Dict[str, Any]]] = None,
            recurring: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Update an existing calendar event in both iCloud and local storage"""
        if event_id not in self.events:
            raise ValueError(f"Event ID {event_id} does not exist")

        event = self.events[event_id]
        old_title = event.title

        # 更新本地事件
        if title is not None:
            event.title = title
        if start_time is not None:
            event.start_time = start_time
        if end_time is not None:
            event.end_time = end_time
        if calendar_type is not None:
            event.calendar_type = calendar_type
        if location is not None:
            event.location = location
        if description is not None:
            event.description = description
        if attendees is not None:
            event.attendees = attendees
        if reminders is not None:
            event.reminders = reminders
        if recurring is not None:
            event.recurring = recurring

        # 同步到 iCloud
        try:
            # 获取 iCloud 事件 ID
            icloud_events = get_apple_calendar_events(
                self.icloud_calendar_name,
                event.start_time - datetime.timedelta(days=1),
                event.end_time + datetime.timedelta(days=1)
            )
            
            icloud_event = None
            for e in icloud_events:
                if e.instance.vevent.summary.value == old_title:
                    icloud_event = e
                    break
            
            if icloud_event:
                success = update_event_in_calendar(
                    self.icloud_calendar_name,
                    icloud_event.instance.vevent.uid.value,
                    event.title,
                    event.start_time,
                    event.end_time
                )
                if not success:
                    print(f"Warning: Failed to sync event update to iCloud: {event.title}")
        except Exception as e:
            print(f"Error syncing update to iCloud: {str(e)}")

        # 保存本地数据
        self.save_data()
        return event.to_dict()

    def delete_event(self, event_id: str) -> bool:
        """Delete a calendar event from both iCloud and local storage"""
        if event_id not in self.events:
            return False

        event = self.events[event_id]
        
        # 从 iCloud 删除
        try:
            icloud_events = get_apple_calendar_events(
                self.icloud_calendar_name,
                event.start_time - datetime.timedelta(days=1),
                event.end_time + datetime.timedelta(days=1)
            )
            
            icloud_event = None
            for e in icloud_events:
                if e.instance.vevent.summary.value == event.title:
                    icloud_event = e
                    break
            
            if icloud_event:
                success = delete_event_from_calendar(
                    self.icloud_calendar_name,
                    icloud_event.instance.vevent.uid.value
                )
                if not success:
                    print(f"Warning: Failed to delete event from iCloud: {event.title}")
        except Exception as e:
            print(f"Error deleting from iCloud: {str(e)}")

        # 从本地存储删除
        del self.events[event_id]
        self.save_data()
        return True

    def get_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get details of a specific event"""
        if event_id not in self.events:
            return None
        return self.events[event_id].to_dict()

    def get_events_by_day(
            self,
            date: datetime.date,
            calendar_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all events for a specific date from both iCloud and local storage"""
        # 获取本地事件
        local_events = []
        for event_id, event in self.events.items():
            event_date = event.start_time.date()
            if event_date == date and (calendar_type is None or event.calendar_type == calendar_type):
                local_events.append(event.to_dict())

        # 获取 iCloud 事件
        try:
            start_datetime = datetime.datetime.combine(date, datetime.time.min)
            end_datetime = datetime.datetime.combine(date, datetime.time.max)
            icloud_events = get_apple_calendar_events(
                self.icloud_calendar_name,
                start_datetime,
                end_datetime
            )
            
            # 转换 iCloud 事件格式
            for event in icloud_events:
                event_dict = {
                    "id": event.instance.vevent.uid.value,
                    "title": event.instance.vevent.summary.value,
                    "start_time": event.instance.vevent.dtstart.value.isoformat(),
                    "end_time": event.instance.vevent.dtend.value.isoformat(),
                    "calendar_type": "icloud",
                    "location": "",
                    "description": "",
                    "attendees": [],
                    "reminders": [],
                    "recurring": {}
                }
                local_events.append(event_dict)
        except Exception as e:
            print(f"Error fetching iCloud events: {str(e)}")

        return local_events

    def get_events_by_date_range(
            self,
            start_date: datetime.date,
            end_date: datetime.date,
            calendar_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get all events within a date range from both iCloud and local storage"""
        # 获取本地事件
        local_events = []
        for event_id, event in self.events.items():
            event_date = event.start_time.date()
            if start_date <= event_date <= end_date and (calendar_type is None or event.calendar_type == calendar_type):
                local_events.append(event.to_dict())

        # 获取 iCloud 事件
        try:
            start_datetime = datetime.datetime.combine(start_date, datetime.time.min)
            end_datetime = datetime.datetime.combine(end_date, datetime.time.max)
            icloud_events = get_apple_calendar_events(
                self.icloud_calendar_name,
                start_datetime,
                end_datetime
            )
            
            # 转换 iCloud 事件格式
            for event in icloud_events:
                event_dict = {
                    "id": event.instance.vevent.uid.value,
                    "title": event.instance.vevent.summary.value,
                    "start_time": event.instance.vevent.dtstart.value.isoformat(),
                    "end_time": event.instance.vevent.dtend.value.isoformat(),
                    "calendar_type": "icloud",
                    "location": "",
                    "description": "",
                    "attendees": [],
                    "reminders": [],
                    "recurring": {}
                }
                local_events.append(event_dict)
        except Exception as e:
            print(f"Error fetching iCloud events: {str(e)}")

        return local_events

    def find_available_slots(
            self,
            date: datetime.date,
            duration_minutes: int,
            start_hour: int = 9,
            end_hour: int = 17,
            calendar_types: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Find available time slots on a specific date"""
        if calendar_types is None:
            calendar_types = self.calendars

        # Get all events for the day
        day_events = []
        for cal_type in calendar_types:
            day_events.extend(self.get_events_by_day(date, cal_type))

        # Sort events by start time
        day_events.sort(key=lambda x: datetime.datetime.fromisoformat(x["start_time"]))

        # Working hours range
        work_start = datetime.datetime.combine(date, datetime.time(start_hour, 0))
        work_end = datetime.datetime.combine(date, datetime.time(end_hour, 0))

        # Find available time slots
        available_slots = []
        current_time = work_start

        for event in day_events:
            event_start = datetime.datetime.fromisoformat(event["start_time"])
            event_end = datetime.datetime.fromisoformat(event["end_time"])

            # Ignore events outside working hours
            if event_end <= work_start or event_start >= work_end:
                continue

            # Check if there's enough free time between current time and event start
            if event_start > current_time:
                free_minutes = (event_start - current_time).total_seconds() / 60
                if free_minutes >= duration_minutes:
                    available_slots.append({
                        "start": current_time.isoformat(),
                        "end": (current_time + datetime.timedelta(minutes=duration_minutes)).isoformat(),
                        "duration_minutes": duration_minutes
                    })

            # Update current time to event end time
            current_time = max(current_time, event_end)

        # Check for available time between last event and end of work day
        if current_time < work_end:
            free_minutes = (work_end - current_time).total_seconds() / 60
            if free_minutes >= duration_minutes:
                available_slots.append({
                    "start": current_time.isoformat(),
                    "end": (current_time + datetime.timedelta(minutes=duration_minutes)).isoformat(),
                    "duration_minutes": duration_minutes
                })

        return available_slots

    def get_todays_events(self, timezone_str: str = "UTC", calendar_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get all events for today in the specified timezone"""
        today = get_current_date(timezone_str)
        return self.get_events_by_day(today, calendar_type)

    def save_data(self):
        """Save calendar data to file"""
        try:
            # Initialize empty data structure if file doesn't exist
            all_user_data = {}
            
            # Load existing data if file exists
            if USER_DATA_FILE.exists():
                try:
                    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:  # Check if file is not empty
                            all_user_data = json.loads(content)
                except json.JSONDecodeError as json_err:
                    print(f"JSON decode error: {str(json_err)}")
                    # If file exists but is invalid JSON, start with empty data
                    all_user_data = {}
                except Exception as read_err:
                    print(f"Error reading file: {str(read_err)}")
                    all_user_data = {}
            
            # Prepare event data for serialization
            serializable_events = {}
            for event_id, event in self.events.items():
                serializable_events[event_id] = event.to_dict()
            
            # Update this user's data
            all_user_data[self.user_id] = {
                'events': serializable_events,
                'calendars': self.calendars
            }
            
            # Create directory if it doesn't exist
            USER_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
            
            # Save all data back to file
            with open(USER_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(all_user_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error saving calendar data: {str(e)}")
            # Create a backup of the corrupted file if it exists
            if USER_DATA_FILE.exists():
                backup_file = USER_DATA_FILE.with_suffix('.json.bak')
                try:
                    import shutil
                    shutil.copy2(USER_DATA_FILE, backup_file)
                    print(f"Created backup of corrupted file at: {backup_file}")
                except Exception as backup_error:
                    print(f"Failed to create backup: {str(backup_error)}")
    
    def load_data(self):
        """Load calendar data from file"""
        try:
            if USER_DATA_FILE.exists():
                try:
                    with open(USER_DATA_FILE, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if not content:  # If file is empty
                            print("Calendar file is empty")
                            return
                        all_user_data = json.loads(content)
                except json.JSONDecodeError as json_err:
                    print(f"Error: Invalid JSON in {USER_DATA_FILE}: {str(json_err)}")
                    return
                except Exception as read_err:
                    print(f"Error reading calendar file: {str(read_err)}")
                    return
                
                # Check if this user has data
                if self.user_id in all_user_data:
                    user_data = all_user_data[self.user_id]
                    
                    # Load calendars
                    self.calendars = user_data.get('calendars', ["personal", "work"])
                    
                    # Load events
                    serialized_events = user_data.get('events', {})
                    for event_id, event_dict in serialized_events.items():
                        try:
                            # Convert ISO format strings back to datetime objects
                            start_time = datetime.datetime.fromisoformat(event_dict['start_time'])
                            end_time = datetime.datetime.fromisoformat(event_dict['end_time'])
                            
                            # Recreate Event object
                            self.events[event_id] = Event(
                                id=event_id,
                                title=event_dict['title'],
                                start_time=start_time,
                                end_time=end_time,
                                calendar_type=event_dict['calendar_type'],
                                location=event_dict['location'],
                                description=event_dict['description'],
                                attendees=event_dict['attendees'],
                                reminders=event_dict['reminders'],
                                recurring=event_dict['recurring']
                            )
                        except Exception as event_error:
                            print(f"Error loading event {event_id}: {str(event_error)}")
                            continue
                else:
                    print(f"No data found for user {self.user_id}")
        except Exception as e:
            print(f"Error loading calendar data: {str(e)}")
            # Start with empty data if loading fails
            self.events = {}


# Create calendar manager instance with default user ID
# This will be replaced with actual user IDs in a multi-user environment
calendar_manager = CalendarManager(user_id="default")


# Define LangChain tools
# 1. Create event tool
class CreateEventInput(BaseModel):
    """Input parameters for creating a calendar event"""
    title: str = Field(..., description="Event title")
    start_time: str = Field(..., description="Event start time (ISO format: YYYY-MM-DDTHH:MM:SS)")
    end_time: str = Field(..., description="Event end time (ISO format: YYYY-MM-DDTHH:MM:SS)")
    calendar_type: str = Field(default="personal", description="Calendar type (personal or work)")
    location: str = Field(default="", description="Event location")
    description: str = Field(default="", description="Event description")
    attendees: List[str] = Field(default_factory=list, description="List of attendees")
    reminders: List[Dict[str, Any]] = Field(default_factory=list, description="Reminder settings")
    recurring: Dict[str, Any] = Field(default_factory=dict, description="Recurring settings")


@tool
def create_event(
        title: str,
        start_time: str,
        end_time: str,
        calendar_type: str = "personal",
        location: str = "",
        description: str = "",
        attendees: List[str] = None,
        reminders: List[Dict[str, Any]] = None,
        recurring: Dict[str, Any] = None
) -> str:
    """
    Create a new calendar event. IMPORTANT: Only use this tool AFTER you have collected ALL necessary information from the user through conversation.
    
    Parameters:
    - title: Event title
    - start_time: Event start time (ISO format: YYYY-MM-DDTHH:MM:SS)
    - end_time: Event end time (ISO format: YYYY-MM-DDTHH:MM:SS)
    - calendar_type: Calendar type ("personal" or "work")
    - location: Event location
    - description: Event description
    - attendees: List of attendees
    - reminders: List of reminder settings
    - recurring: Recurring event settings
    
    Returns:
    - JSON string containing details of the newly created event
    """
    try:
        start_dt = datetime.datetime.fromisoformat(start_time)
        end_dt = datetime.datetime.fromisoformat(end_time)

        event = calendar_manager.create_event(
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            calendar_type=calendar_type,
            location=location,
            description=description,
            attendees=attendees,
            reminders=reminders,
            recurring=recurring
        )

        return json.dumps(event, indent=2)
    except Exception as e:
        return f"Failed to create event: {str(e)}"


# 2. Update event tool
class UpdateEventInput(BaseModel):
    """Input parameters for updating a calendar event"""
    event_id: str = Field(..., description="ID of the event to update")
    title: Optional[str] = Field(None, description="Event title")
    start_time: Optional[str] = Field(None, description="Event start time (ISO format)")
    end_time: Optional[str] = Field(None, description="Event end time (ISO format)")
    calendar_type: Optional[str] = Field(None, description="Calendar type")
    location: Optional[str] = Field(None, description="Event location")
    description: Optional[str] = Field(None, description="Event description")
    attendees: Optional[List[str]] = Field(None, description="List of attendees")
    reminders: Optional[List[Dict[str, Any]]] = Field(None, description="Reminder settings")
    recurring: Optional[Dict[str, Any]] = Field(None, description="Recurring settings")


@tool
def update_event(
        event_id: str,
        title: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        calendar_type: Optional[str] = None,
        location: Optional[str] = None,
        description: Optional[str] = None,
        attendees: Optional[List[str]] = None,
        reminders: Optional[List[Dict[str, Any]]] = None,
        recurring: Optional[Dict[str, Any]] = None
) -> str:
    """
    Update an existing calendar event. IMPORTANT: Only use this tool AFTER you have collected ALL necessary information from the user through conversation.
    
    Parameters:
    - event_id: ID of the event to update
    - title: Updated title (optional)
    - start_time: Updated start time (ISO format, optional)
    - end_time: Updated end time (ISO format, optional)
    - calendar_type: Updated calendar type (optional)
    - location: Updated location (optional)
    - description: Updated description (optional)
    - attendees: Updated list of attendees (optional)
    - reminders: Updated reminder settings (optional)
    - recurring: Updated recurring settings (optional)
    
    Returns:
    - JSON string containing details of the updated event
    """
    try:
        start_dt = None
        if start_time:
            start_dt = datetime.datetime.fromisoformat(start_time)

        end_dt = None
        if end_time:
            end_dt = datetime.datetime.fromisoformat(end_time)

        event = calendar_manager.update_event(
            event_id=event_id,
            title=title,
            start_time=start_dt,
            end_time=end_dt,
            calendar_type=calendar_type,
            location=location,
            description=description,
            attendees=attendees,
            reminders=reminders,
            recurring=recurring
        )

        return json.dumps(event, indent=2)
    except Exception as e:
        return f"Failed to update event: {str(e)}"


# 3. Delete event tool
@tool
def delete_event(event_id: str) -> str:
    """
    Delete a calendar event.
    
    Parameters:
    - event_id: ID of the event to delete
    
    Returns:
    - Success or failure message
    """
    try:
        result = calendar_manager.delete_event(event_id)
        if result:
            return f"Event {event_id} successfully deleted"
        else:
            return f"Event {event_id} does not exist"
    except Exception as e:
        return f"Failed to delete event: {str(e)}"


# 4. Get daily schedule tool
@tool
def get_daily_schedule(date_str: str, calendar_type: Optional[str] = None) -> str:
    """
    Get schedule for a specific date.
    
    Parameters:
    - date_str: Date string (format: YYYY-MM-DD)
    - calendar_type: Calendar type ("personal", "work", or None for all)
    
    Returns:
    - JSON string containing a list of events for the day
    """
    try:
        date = datetime.date.fromisoformat(date_str)
        events = calendar_manager.get_events_by_day(date, calendar_type)
        return json.dumps(events, indent=2)
    except Exception as e:
        return f"Failed to get schedule: {str(e)}"


# 5. Get weekly/monthly view tool
@tool
def get_date_range_schedule(
        start_date_str: str,
        end_date_str: str,
        calendar_type: Optional[str] = None
) -> str:
    """
    Get schedule for a date range.
    
    Parameters:
    - start_date_str: Start date (format: YYYY-MM-DD)
    - end_date_str: End date (format: YYYY-MM-DD)
    - calendar_type: Calendar type ("personal", "work", or None for all)
    
    Returns:
    - JSON string containing a list of events in the date range
    """
    try:
        start_date = datetime.date.fromisoformat(start_date_str)
        end_date = datetime.date.fromisoformat(end_date_str)
        events = calendar_manager.get_events_by_date_range(start_date, end_date, calendar_type)
        return json.dumps(events, indent=2)
    except Exception as e:
        return f"Failed to get date range schedule: {str(e)}"


# 6. Find available time slots tool
@tool
def find_available_time_slots(
        date_str: str,
        duration_minutes: int,
        start_hour: int = 9,
        end_hour: int = 17,
        calendar_types: List[str] = None
) -> str:
    """
    Find available time slots on a specific date.
    
    Parameters:
    - date_str: Date string (format: YYYY-MM-DD)
    - duration_minutes: Required duration in minutes
    - start_hour: Start of working day hour (default: 9)
    - end_hour: End of working day hour (default: 17)
    - calendar_types: List of calendar types to check (default: all)
    
    Returns:
    - JSON string containing a list of available time slots
    """
    try:
        date = datetime.date.fromisoformat(date_str)
        slots = calendar_manager.find_available_slots(
            date=date,
            duration_minutes=duration_minutes,
            start_hour=start_hour,
            end_hour=end_hour,
            calendar_types=calendar_types
        )
        return json.dumps(slots, indent=2)
    except Exception as e:
        return f"Failed to find available time slots: {str(e)}"


# 7. Get event details tool
@tool
def get_event_details(event_id: str) -> str:
    """
    Get details of a specific event.
    
    Parameters:
    - event_id: Event ID
    
    Returns:
    - JSON string containing event details
    """
    try:
        event = calendar_manager.get_event(event_id)
        if event:
            return json.dumps(event, indent=2)
        else:
            return f"Event {event_id} does not exist"
    except Exception as e:
        return f"Failed to get event details: {str(e)}"


# 8. Get current date and time tool
@tool
def get_current_datetime_info(timezone: str = "UTC") -> str:
    """
    Get the current date and time information.
    
    Parameters:
    - timezone: Timezone name (default: "UTC")
    
    Returns:
    - JSON string containing current date and time information
    """
    try:
        now = get_current_datetime(timezone)
        current_info = {
            "datetime": now.isoformat(),
            "date": now.date().isoformat(),
            "time": now.time().isoformat(),
            "year": now.year,
            "month": now.month,
            "day": now.day,
            "hour": now.hour,
            "minute": now.minute,
            "second": now.second,
            "timezone": timezone,
            "weekday": now.strftime("%A")
        }
        return json.dumps(current_info, indent=2)
    except Exception as e:
        return f"Failed to get current datetime info: {str(e)}"


# 9. Get today's schedule tool
@tool
def get_todays_schedule(timezone: str = "UTC", calendar_type: Optional[str] = None) -> str:
    """
    Get schedule for today.
    
    Parameters:
    - timezone: Timezone name (default: "UTC")
    - calendar_type: Calendar type ("personal", "work", or None for all)
    
    Returns:
    - JSON string containing a list of events for today
    """
    try:
        today = get_current_date(timezone)
        events = calendar_manager.get_events_by_day(today, calendar_type)

        # Add current date information
        result = {
            "date": today.isoformat(),
            "weekday": datetime.datetime.combine(today, datetime.time()).strftime("%A"),
            "events": events
        }

        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Failed to get today's schedule: {str(e)}"


# 10. Send email tool
class SendEmailInput(BaseModel):
    """Input parameters for sending an email"""
    receiver_emails: str = Field(..., description="Comma-separated list of receiver email addresses")
    subject: str = Field(..., description="Email subject")
    content: str = Field(..., description="Email content")
    attachments: Optional[List[str]] = Field(default=None, description="List of attachment file paths")

@tool
def send_email(receiver_emails: str, subject: str, content: str, attachments: Optional[List[str]] = None) -> str:
    """
    Send an email to one or multiple recipients.
    
    Parameters:
    - receiver_emails: Comma-separated list of receiver email addresses
    - subject: Email subject
    - content: Email content
    - attachments: Optional list of attachment file paths
    
    Returns:
    - JSON string containing success and failure information
    """
    try:
        success_list, fail_list = send_bulk_email(
            receiver_emails=receiver_emails,
            subject=subject,
            content=content,
            attachments=attachments
        )
        
        result = {
            "success": {
                "count": len(success_list),
                "emails": success_list
            },
            "failed": {
                "count": len(fail_list),
                "emails": fail_list
            }
        }
        
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Failed to send email: {str(e)}"


# Create tools list
tools = [
    create_event,
    update_event,
    delete_event,
    get_daily_schedule,
    get_date_range_schedule,
    find_available_time_slots,
    get_event_details,
    get_current_datetime_info,
    get_todays_schedule,
    send_email  # Add the new email tool
]

# Define system prompt for the ChronosLink Smart Scheduler Assistant
system_prompt = f"""You are ChronosLink, a Smart Scheduler Assistant designed to help users manage their calendars efficiently.
Today's date and time is {datetime.datetime.now()}. Use this as a reference for scheduling tasks.

You can help users with the following tasks:
1. Creating, updating, and deleting calendar events
2. Finding available time slots for meetings and appointments
3. Retrieving schedule information for specific days or date ranges
4. Managing events across multiple calendars (personal and work)
5. Setting up reminders and recurring events

When handling date/time information:
- Always use ISO format (YYYY-MM-DDTHH:MM:SS) when working with tools
- Help users by converting natural language time references to proper formats
- Make reasonable assumptions about time when not explicitly provided (e.g., business hours, duration)
- Be aware of time zones and clarify them when necessary
- You can use the get_current_datetime_info tool to get the current date and time in any timezone
- You can use the get_todays_schedule tool to directly access today's calendar events
- If today or eg. "this afternoon" is mentioned, use the get_todays_schedule tool to get the events for today or get the time using get_current_datetime_info

IMPORTANT: For event creation and updates, ensure that you have all required information before using the appropriate tool:
- If the user provides incomplete information for creating an event, DO NOT guess or immediately use the create_event tool.
- Instead, ask follow-up questions to gather necessary details such as:
  * Event title (interpret the title from the user prompt, eg. "Schedule a meeting with Sarah tomorrow at 3 PM" -> "Meeting with Sarah")
  * Date and time (if missing or unclear)
  * Duration (if missing, if not provided, assume 1 hour)
  * Location (if not given, skip it)
- Only use the create_event tool when you have collected all the essential information. But do not ask for the information if the user has already provided it and if that information is not necessary for the tool call.
- For recurring events, ask about the frequency and end date if not provided.
Your response should be concise and to the point. Should not be in bullet points.
Your primary goal is to make scheduling as easy as possible by understanding user requests and taking appropriate actions. Maintain a conversational flow to ensure all necessary details are collected before finalizing any calendar operation.

Use the provided tools to interact with the calendar system and maintain a friendly, helpful tone in all communications.
"""


# Create LangGraph agent with the defined tools
# Define the state for the LangGraph
class AgentState(TypedDict):
    """State definition for the calendar agent"""
    messages: Annotated[list, add_messages]
    # Add more state fields if needed


# Create a react agent using the LLM, tools, and system message
agent = create_react_agent(llm, tools)

# Create ToolNode for executing tools
tool_node = ToolNode(tools)


# Define state handling functions
def should_continue(state: AgentState) -> bool:
    """Check if the agent should continue running based on the last AI message"""
    messages = state.get("messages", [])
    if not messages:
        return False

    last_message = messages[-1]
    # If the last message is from the AI and doesn't contain a tool call, we're done
    # This allows the AI to ask follow-up questions without immediately calling a tool
    if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
        return False

    return True


# Build the LangGraph
def build_graph():
    # Initialize the state graph
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("agent", agent)
    workflow.add_node("tools", tool_node)

    # Add edges to connect the nodes
    # Start -> agent
    workflow.set_entry_point("agent")

    # agent -> tools (when tools are called)
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            True: "tools",  # If the agent calls a tool, go to tools
            False: END  # If the agent is done, end
        }
    )

    # tools -> agent (always return to agent after tool execution)
    workflow.add_edge("tools", "agent")

    # Compile the graph with memory saver for conversation persistence
    return workflow.compile(checkpointer=memory_saver)


# Build the agent graph
agent_executor = build_graph()


# Chat function for handling user messages
def chat(message, history, thread_id, user_id):
    # Create a new thread id if not already present
    if not thread_id:
        thread_id = str(uuid.uuid4())

    # Set up the config using the session-specific thread id and user id
    config = {"configurable": {"thread_id": f"{user_id}_{thread_id}", "user_id": user_id}}

    # Append the user's message and a placeholder for the bot's response to the chat history
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
    response_index = len(history) - 1  # Index of the bot's response in history

    full_response = ""
    tool_calls = []

    # Stream the output from the backend in chunks
    for chunk in agent_executor.stream(
            {"messages": [HumanMessage(content=message)]},
            config,
            stream_mode="values",
    ):
        if "messages" in chunk and chunk["messages"] and isinstance(chunk["messages"][-1], AIMessage):
            ai_message = chunk["messages"][-1]
            full_response = ai_message.content if ai_message.content else full_response

            # Track tool calls
            if hasattr(ai_message, "tool_calls") and ai_message.tool_calls:
                for tool_call in ai_message.tool_calls:
                    tool_calls.append({
                        "tool": tool_call.name,
                        "parameters": tool_call.args,
                        "timestamp": datetime.datetime.now().isoformat()
                    })

            # Update the last chat tuple with the new partial response
            history[response_index] = {"role": "assistant", "content": full_response}
            # Yield the updated chat history and tool calls
            yield history, thread_id, tool_calls


# Build the Gradio interface
with gr.Blocks() as demo:
    # State variables
    thread_state = gr.State()  # Holds the unique thread id across user interactions
    user_id_state = gr.State("default")  # Holds the user ID
    voice_output_enabled = gr.State(False)  # State to track if voice output is enabled
    tool_calls_history = gr.State([])  # State to store tool calls history

    # Header
    gr.Markdown("# ChronosLink: Smart Scheduler Assistant")
    gr.Markdown(
        "Manage your calendar with natural language. Ask me to schedule events, find free time, or check your agenda.")

    # User authentication section
    with gr.Row():
        user_id_input = gr.Textbox(
            placeholder="Enter your user ID (optional)",
            label="User ID",
            value="default"
        )
        login_btn = gr.Button("Login")

    # Create tabs
    with gr.Tabs() as tabs:
        # Chat Tab
        with gr.Tab("Chat"):
            # Voice options
            with gr.Row():
                enable_voice_output = gr.Checkbox(label="Enable Voice Output", value=False)
                voice_type = gr.Dropdown(
                    choices=["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                    label="Voice Type",
                    value="alloy"
                )

            # Chat interface
            chatbot = gr.Chatbot(height=500, type="messages")

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message here (e.g., 'Schedule a meeting with Sarah tomorrow at 3 PM')",
                    scale=4
                )
                send_btn = gr.Button("Send", scale=1)

            # Voice input
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Voice Input"
                )
                send_audio_btn = gr.Button("Send Voice Message")

            clear_btn = gr.Button("Clear Chat")

            # For displaying AI voice output
            ai_audio_output = gr.Audio(label="AI Voice Response", visible=True, autoplay=True)

        # Calendar Tab
        with gr.Tab("Calendar"):
            with gr.Row():
                calendar_type = gr.Dropdown(
                    choices=["all", "personal", "work"],
                    label="Calendar Type",
                    value="all"
                )

            with gr.Row():
                start_date = gr.Textbox(
                    label="Start Date (YYYY-MM-DD)",
                    placeholder="YYYY-MM-DD",
                    value=datetime.datetime.now().strftime("%Y-%m-%d")
                )
                end_date = gr.Textbox(
                    label="End Date (YYYY-MM-DD)",
                    placeholder="YYYY-MM-DD",
                    value=(datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d")
                )

            calendar_view = gr.Markdown(label="Calendar Events")


            def format_event(event):
                start_time = datetime.datetime.fromisoformat(event["start_time"])
                end_time = datetime.datetime.fromisoformat(event["end_time"])
                return f"""
### {event['title']}
- **Time**: {start_time.strftime('%Y-%m-%d %H:%M')} - {end_time.strftime('%H:%M')}
- **Calendar**: {event['calendar_type']}
- **Location**: {event['location'] if event['location'] else 'No location specified'}
- **Description**: {event['description'] if event['description'] else 'No description'}
- **Attendees**: {', '.join(event['attendees']) if event['attendees'] else 'No attendees'}
---
"""


            def update_calendar_view(start_str, end_str, cal_type):
                try:
                    start_date = datetime.datetime.strptime(start_str, "%Y-%m-%d").date()
                    end_date = datetime.datetime.strptime(end_str, "%Y-%m-%d").date()

                    if cal_type == "all":
                        cal_type = None

                    events = calendar_manager.get_events_by_date_range(start_date, end_date, cal_type)

                    if not events:
                        return "No events found in the selected date range."

                    # Sort events by start time
                    events.sort(key=lambda x: x["start_time"])

                    # Group events by date
                    events_by_date = {}
                    for event in events:
                        date = datetime.datetime.fromisoformat(event["start_time"]).date()
                        if date not in events_by_date:
                            events_by_date[date] = []
                        events_by_date[date].append(event)

                    # Format the output
                    output = ""
                    for date in sorted(events_by_date.keys()):
                        output += f"\n## {date.strftime('%Y-%m-%d')} ({date.strftime('%A')})\n"
                        for event in events_by_date[date]:
                            output += format_event(event)

                    return output
                except ValueError:
                    return "Error: Invalid date format. Please use YYYY-MM-DD"


            date_inputs = [start_date, end_date, calendar_type]
            for input_component in date_inputs:
                input_component.change(
                    update_calendar_view,
                    inputs=date_inputs,
                    outputs=[calendar_view]
                )

        # Tool Calls Tab
        with gr.Tab("Tool Calls"):
            tool_calls_display = gr.Markdown(label="Recent Tool Calls")

            def format_tool_call(tool_call):
                try:
                    params = json.dumps(tool_call['parameters'], indent=2, ensure_ascii=False)
                    return f"""
### {tool_call['tool']}
- **Time**: {datetime.datetime.fromisoformat(tool_call['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
- **Parameters**:
```json
{params}
```
---
"""
                except Exception as e:
                    print(f"Error formatting tool call: {str(e)}")
                    return f"Error displaying tool call: {str(e)}"

            def update_tool_calls(tool_calls):
                if not tool_calls:
                    return "No tool calls recorded yet."

                try:
                    # Sort tool calls by timestamp
                    tool_calls.sort(key=lambda x: x['timestamp'], reverse=True)

                    output = ""
                    for tool_call in tool_calls:
                        output += format_tool_call(tool_call)
                    return output
                except Exception as e:
                    print(f"Error updating tool calls display: {str(e)}")
                    return f"Error displaying tool calls: {str(e)}"

            # Update the tool calls display whenever tool_calls_history changes
            tool_calls_history.change(
                update_tool_calls,
                inputs=[tool_calls_history],
                outputs=[tool_calls_display]
            )


    # Update voice output state
    def update_voice_state(enabled):
        return enabled


    enable_voice_output.change(
        update_voice_state,
        inputs=[enable_voice_output],
        outputs=[voice_output_enabled]
    )


    # Process audio input
    def process_audio_input(audio_path):
        if not audio_path:
            return "Please record a voice message first."

        transcribed_text, success = speech_to_text(audio_path)
        if success and transcribed_text:
            return transcribed_text
        else:
            return "Sorry, I couldn't understand the audio. Please try again."


    audio_input.change(
        process_audio_input,
        inputs=[audio_input],
        outputs=[msg_input]
    )


    # Login function to switch users
    def login(user_id):
        # Create a new calendar manager for this user
        global calendar_manager
        calendar_manager = CalendarManager(user_id=user_id)

        # Create a new thread ID for this user session
        new_thread_id = str(uuid.uuid4())

        # Return the new user ID and a welcome message
        return user_id, [{"role": "system", "content": f"Logged in as: {user_id}"}], new_thread_id


    # Connect login button
    login_btn.click(
        login,
        inputs=[user_id_input],
        outputs=[user_id_state, chatbot, thread_state],
    )


    # Chat function with voice output and tool calls tracking
    def chat_with_voice(message, history, thread_id, user_id, voice_enabled, voice, tool_calls):
        # Create base generator
        chat_generator = chat(message, history, thread_id, user_id)

        # Process each response
        for history_update, thread_id_update, tool_calls_update in chat_generator:
            # If this is the final response and voice is enabled
            if voice_enabled:
                last_response = history_update[-1]["content"]
                # Generate voice for the last AI response
                audio_data, success = text_to_speech(last_response, voice)
                if success:
                    # Save to temporary file
                    audio_path = save_audio_to_temp_file(audio_data)
                    if audio_path:
                        # Return everything including updated tool calls and autoplay the audio
                        yield history_update, thread_id_update, audio_path, tool_calls_update
                        # Clean up the temporary file after a delay
                        cleanup_temp_file(audio_path)
                        continue

            # If voice was disabled or failed, just return the text response
            yield history_update, thread_id_update, None, tool_calls_update


    # The click event for text input
    send_btn.click(
        chat_with_voice,
        inputs=[msg_input, chatbot, thread_state, user_id_state, voice_output_enabled, voice_type, tool_calls_history],
        outputs=[chatbot, thread_state, ai_audio_output, tool_calls_history],
    )

    # The click event for voice input
    send_audio_btn.click(
        chat_with_voice,
        inputs=[msg_input, chatbot, thread_state, user_id_state, voice_output_enabled, voice_type, tool_calls_history],
        outputs=[chatbot, thread_state, ai_audio_output, tool_calls_history],
    )

    # Allow sending messages with Enter key
    msg_input.submit(
        chat_with_voice,
        inputs=[msg_input, chatbot, thread_state, user_id_state, voice_output_enabled, voice_type, tool_calls_history],
        outputs=[chatbot, thread_state, ai_audio_output, tool_calls_history],
    )

    # Clear input after sending
    send_btn.click(lambda: "", None, msg_input)
    msg_input.submit(lambda: "", None, msg_input)

    # Clear the chat history
    clear_btn.click(
        lambda: ([], str(uuid.uuid4()), None, []),
        None,
        [chatbot, thread_state, ai_audio_output, tool_calls_history]
    )

# Launch the Gradio app
if __name__ == "__main__":
    print(system_prompt)
    demo.launch()
