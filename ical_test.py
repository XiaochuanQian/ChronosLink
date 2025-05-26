import os
import caldav
from caldav.elements import dav, cdav
import datetime
import json
import pathlib

# Load configuration
def load_config():
    config_path = pathlib.Path("config.json")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# Load credentials from config
config = load_config()
APPLE_ID = config.get('calendar', {}).get('apple_id')
APPLE_PASSWORD = config.get('calendar', {}).get('apple_password')
CALENDAR_URL = config.get('calendar', {}).get('calendar_url')

# APPLE_ID = "xiaochuanqian@icloud.com"
# APPLE_PASSWORD = "pyle-zmcg-fqig-ihpp"
# CALENDAR_URL = 'https://caldav.icloud.com'  # We'll use the base URL


def get_caldav_client():
    return caldav.DAVClient(url=CALENDAR_URL, username=APPLE_ID, password=APPLE_PASSWORD)


def discover_caldav_calendars():
    try:
        client = get_caldav_client()
        principal = client.principal()
        print(f"Principal URL: {principal.url}")

        calendars = principal.calendars()

        if calendars:
            print("Available calendars:")
            for calendar in calendars:
                print(f"- {calendar.name} (URL: {calendar.url})")
        else:
            print("No calendars found.")

        return CALENDAR_URL

    except caldav.lib.error.AuthorizationError as e:
        print(f"Authorization failed: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None


def get_apple_calendar_events(calendar_name, start_date, end_date):
    client = get_caldav_client()
    principal = client.principal()
    calendars = principal.calendars()

    calendar = next((cal for cal in calendars if cal.name == calendar_name), None)

    if calendar:
        events = calendar.date_search(start=start_date, end=end_date)
        return events
    else:
        print(f"Calendar '{calendar_name}' not found.")
        return None


def clean_ical_data(event):
    """清理和标准化 iCalendar 数据"""
    if hasattr(event, 'instance') and hasattr(event.instance, 'vevent'):
        vevent = event.instance.vevent
        # 确保只有一个 DTSTAMP
        if hasattr(vevent, 'dtstamp') and len(vevent.dtstamp_list) > 1:
            # 保留最新的 DTSTAMP
            latest_dtstamp = max(vevent.dtstamp_list, key=lambda x: x.value)
            vevent.dtstamp_list = [latest_dtstamp]
    return event


def add_event_to_calendar(calendar_name, summary, start_time, end_time, rrule=None):
    client = get_caldav_client()
    principal = client.principal()
    calendars = principal.calendars()

    calendar = next((cal for cal in calendars if cal.name == calendar_name), None)

    if calendar:
        try:
            # 创建基本事件
            event = calendar.save_event(
                dtstart=start_time,
                dtend=end_time,
                summary=summary,
                rrule=rrule
            )
            # 清理和标准化 iCalendar 数据
            event = clean_ical_data(event)
            event.save()
            return True
        except Exception as e:
            print(f"Error creating event: {str(e)}")
            return False
    else:
        print(f"Calendar '{calendar_name}' not found.")
        return False


def update_event_in_calendar(calendar_name, event_uid, summary, start_time, end_time, rrule=None):
    client = get_caldav_client()
    principal = client.principal()
    calendars = principal.calendars()

    calendar = next((cal for cal in calendars if cal.name == calendar_name), None)

    if calendar:
        try:
            event = calendar.event(event_uid)
            event.load()
            
            # 更新事件属性
            event.instance.vevent.summary.value = summary
            event.instance.vevent.dtstart.value = start_time
            event.instance.vevent.dtend.value = end_time
            
            # 更新重复规则
            if rrule:
                if hasattr(event.instance.vevent, 'rrule'):
                    event.instance.vevent.rrule.value = rrule
                else:
                    event.instance.vevent.add('rrule', rrule)
            
            # 清理和标准化 iCalendar 数据
            event = clean_ical_data(event)
            
            # 保存更新
            event.save()
            return True
        except Exception as e:
            print(f"Error updating event: {str(e)}")
            return False
    else:
        print(f"Calendar '{calendar_name}' not found.")
        return False


def delete_event_from_calendar(calendar_name, event_uid):
    client = get_caldav_client()
    principal = client.principal()
    calendars = principal.calendars()

    calendar = next((cal for cal in calendars if cal.name == calendar_name), None)

    if calendar:
        event = calendar.event(event_uid)
        event.delete()
        return True
    else:
        print(f"Calendar '{calendar_name}' not found.")
        return False


def list_calendars():
    client = get_caldav_client()
    principal = client.principal()
    calendars = principal.calendars()

    return [{'name': cal.name, 'url': cal.url} for cal in calendars]


# Example usage
if __name__ == "__main__":
    caldav_url = discover_caldav_calendars()
    if caldav_url:
        print(f"\niCloud CalDAV is accessible. Base URL: {caldav_url}")

        # List calendars
        calendars = list_calendars()
        print("\nCalendars:")
        for cal in calendars:
            print(f"- {cal['name']} ({cal['url']})")

        # Example: Get events for a specific calendar
        calendar_name = "calendar_name"
        start_date = datetime.datetime.now()
        end_date = start_date + datetime.timedelta(days=7)
        events = get_apple_calendar_events(calendar_name, start_date, end_date)
        if events:
            print(f"\nEvents in '{calendar_name}' for the next 7 days:")
            for event in events:
                print(f"- {event.instance.vevent.summary.value}")

        # Example: Add an event
        add_event_to_calendar(calendar_name, "[Testing] New Event", start_date, end_date)

        # Note: For update and delete operations, you'd need the event's UID,
        # which you can get from the event objects returned by get_apple_calendar_events
    else:
        print("\nFailed to access iCloud CalDAV.")