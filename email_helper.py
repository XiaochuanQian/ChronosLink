import zmail
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
sender_email = config.get('email', {}).get('sender_email')
sender_password = config.get('email', {}).get('sender_password')

def send_bulk_email(receiver_emails, subject, content, attachments=None):
    """
    批量发送邮件的函数
    
    参数:
        receiver_emails: 收件人邮箱列表或逗号分隔的邮箱字符串
        subject: 邮件主题
        content: 邮件内容
        attachments: 附件列表（可选）
    
    返回:
        tuple: (成功发送的邮箱列表, 失败的邮箱列表)
    """
    # 发件人信息
    sender_email = "linqingwen23@shphschool.com"
    sender_password = "4LsCjvR9Jg6ahDCS"
    
    # 如果输入是字符串，转换为列表
    if isinstance(receiver_emails, str):
        receiver_emails = [email.strip() for email in receiver_emails.split(',')]
    
    # 创建邮件服务器
    server = zmail.server(sender_email, sender_password)
    
    # 邮件内容
    mail_content = {
        'subject': subject,
        'content_text': content,
        'attachments': attachments
    }
    
    success_list = []
    fail_list = []
    
    # 发送邮件给所有收件人
    for receiver_email in receiver_emails:
        try:
            server.send_mail(receiver_email.strip(), mail_content)
            print(f"邮件发送成功！收件人：{receiver_email.strip()}")
            success_list.append(receiver_email.strip())
        except Exception as e:
            print(f"邮件发送失败：{receiver_email.strip()} - {str(e)}")
            fail_list.append(receiver_email.strip())
    
    return success_list, fail_list

def main():
    # 收件人信息
    print("请输入收件人邮箱（多个邮箱请用逗号分隔）：")
    receiver_emails = input()
    
    # 邮件内容
    subject = input("请输入邮件主题: ")
    content = input("请输入邮件内容: ")
    
    # 是否添加附件
    add_attachment = input("是否需要添加附件？(y/n): ").lower()
    attachments = None
    
    if add_attachment == 'y':
        attachment_path = input("请输入附件路径: ")
        attachments = [attachment_path]
    
    # 发送邮件
    success_list, fail_list = send_bulk_email(
        receiver_emails=receiver_emails,
        subject=subject,
        content=content,
        attachments=attachments
    )
    
    # 打印发送结果统计
    print("\n发送结果统计：")
    print(f"成功发送：{len(success_list)} 封")
    print(f"发送失败：{len(fail_list)} 封")

if __name__ == "__main__":
    main()