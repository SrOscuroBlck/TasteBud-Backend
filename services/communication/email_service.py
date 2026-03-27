import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from uuid import UUID
from config.settings import settings
from utils.logger import setup_logger

logger = setup_logger(__name__)


class EmailService:
    def __init__(self):
        self.smtp_host = settings.SMTP_HOST
        self.smtp_port = settings.SMTP_PORT
        self.smtp_user = settings.SMTP_USER
        self.smtp_password = settings.SMTP_PASSWORD
        self.from_email = settings.SMTP_FROM_EMAIL
        self.from_name = settings.SMTP_FROM_NAME
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = f"{self.from_name} <{self.from_email}>"
            message["To"] = to_email
            
            if text_body:
                part1 = MIMEText(text_body, "plain")
                message.attach(part1)
            
            part2 = MIMEText(html_body, "html")
            message.attach(part2)
            
            server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(self.smtp_user, self.smtp_password)
            server.sendmail(self.from_email, [to_email], message.as_string())
            server.quit()
            
            logger.info("Email sent successfully", extra={"to": to_email, "subject": subject})
            return True
            
        except Exception as e:
            logger.error(
                "Failed to send email",
                extra={"to": to_email, "error": str(e), "error_type": type(e).__name__},
                exc_info=True
            )
            return False
    
    def send_otp_code(self, to_email: str, code: str) -> bool:
        subject = "Your TasteBud Login Code"

        text_body = f"""
    Your TasteBud verification code is: {code}

    This code will expire in 10 minutes.

    If you didn't request this code, please ignore this email.

    - The TasteBud Team
    """

        html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; background-color: #111111; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #111111;">
            <tr>
                <td align="center" style="padding: 20px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width: 480px; background-color: #111111;">
                        <tr>
                            <td style="padding: 56px 40px 20px 40px; text-align: center;">
                                <h1 style="margin: 0 0 8px 0; font-size: 38px; font-weight: 700; letter-spacing: -0.5px;">
                                    <span style="color: #E84A3C;">Taste</span><span style="color: #FF6B4A;">Bud</span>
                                </h1>
                                <div style="width: 48px; height: 3px; background: linear-gradient(90deg, #E84A3C, #FF6B4A); margin: 0 auto 40px auto; border-radius: 2px;"></div>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px; text-align: center;">
                                <h2 style="margin: 0 0 8px 0; font-size: 22px; font-weight: 700; color: #F5F5F5; letter-spacing: -0.3px;">Your personal login code</h2>
                                <p style="margin: 0 0 32px 0; font-size: 15px; color: #6B7280; line-height: 1.5;">Enter this code to access your personalized food recommendations</p>
                            </td>
                        </tr>
                        <!-- Code box -->
                        <tr>
                            <td style="padding: 0 32px;">
                                <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                                    <tr>
                                        <td style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 32px 20px; text-align: center;">
                                            <div style="font-family: 'Courier New', monospace; font-size: 40px; font-weight: 700; letter-spacing: 12px; color: #E84A3C; margin: 0;">{code}</div>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        <!-- Expiry -->
                        <tr>
                            <td style="padding: 24px 40px 0 40px; text-align: center;">
                                <p style="margin: 0; font-size: 14px; color: #6B7280;">Expires in <span style="color: #F5F5F5; font-weight: 600;">10 minutes</span></p>
                            </td>
                        </tr>
                        <!-- Footer -->
                        <tr>
                            <td style="padding: 40px 40px 16px 40px; text-align: center;">
                                <div style="width: 100%; height: 1px; background: rgba(255, 255, 255, 0.06); margin-bottom: 24px;"></div>
                                <p style="margin: 0 0 4px 0; font-size: 12px; color: #4B5563; line-height: 1.5;">If you didn't request this code, you can safely ignore this email.</p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px 48px 40px; text-align: center;">
                                <p style="margin: 0; font-size: 13px; font-weight: 500; color: #E84A3C;">Made with ❤️ for food lovers</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

        return self.send_email(to_email, subject, html_body, text_body)


    
    def send_magic_link(self, to_email: str, token: str) -> bool:
        magic_link = f"{settings.FRONTEND_URL}/auth/verify?token={token}"
        subject = "Your TasteBud Magic Link"

        text_body = f"""
    Click the link below to log in to TasteBud:

    {magic_link}

    This link will expire in 10 minutes.

    If you didn't request this link, please ignore this email.

    - The TasteBud Team
    """

        html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; background-color: #111111; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #111111;">
            <tr>
                <td align="center" style="padding: 20px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width: 480px; background-color: #111111;">
                        <tr>
                            <td style="padding: 56px 40px 20px 40px; text-align: center;">
                                <h1 style="margin: 0 0 8px 0; font-size: 38px; font-weight: 700; letter-spacing: -0.5px;">
                                    <span style="color: #E84A3C;">Taste</span><span style="color: #FF6B4A;">Bud</span>
                                </h1>
                                <div style="width: 48px; height: 3px; background: linear-gradient(90deg, #E84A3C, #FF6B4A); margin: 0 auto 40px auto; border-radius: 2px;"></div>
                            </td>
                        </tr>
                        <!-- Heading -->
                        <tr>
                            <td style="padding: 0 40px; text-align: center;">
                                <h2 style="margin: 0 0 8px 0; font-size: 22px; font-weight: 700; color: #F5F5F5; letter-spacing: -0.3px;">Your magic link is ready</h2>
                                <p style="margin: 0 0 36px 0; font-size: 15px; color: #6B7280; line-height: 1.5;">Tap the button below to access your personalized food recommendations</p>
                            </td>
                        </tr>
                        <!-- CTA Button -->
                        <tr>
                            <td style="padding: 0 40px; text-align: center;">
                                <table role="presentation" cellpadding="0" cellspacing="0" style="margin: 0 auto;">
                                    <tr>
                                        <td style="border-radius: 14px; background: linear-gradient(135deg, #E84A3C 0%, #FF6B4A 100%); box-shadow: 0 8px 24px rgba(232, 74, 60, 0.25);">
                                            <a href="{magic_link}" target="_blank" style="display: inline-block; padding: 16px 48px; font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; font-weight: 700; color: #FFFFFF; text-decoration: none; letter-spacing: -0.2px;">Log In to TasteBud</a>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        <!-- Fallback link -->
                        <tr>
                            <td style="padding: 20px 40px 0 40px; text-align: center;">
                                <p style="margin: 0; font-size: 12px; color: #4B5563;">Or copy this link:</p>
                                <p style="margin: 4px 0 0 0; font-size: 12px; word-break: break-all;"><a href="{magic_link}" style="color: #E84A3C; text-decoration: underline;">{magic_link}</a></p>
                            </td>
                        </tr>
                        <!-- Expiry -->
                        <tr>
                            <td style="padding: 28px 40px 0 40px; text-align: center;">
                                <p style="margin: 0; font-size: 14px; color: #6B7280;">Expires in <span style="color: #F5F5F5; font-weight: 600;">10 minutes</span></p>
                            </td>
                        </tr>
                        <!-- Footer -->
                        <tr>
                            <td style="padding: 40px 40px 16px 40px; text-align: center;">
                                <div style="width: 100%; height: 1px; background: rgba(255, 255, 255, 0.06); margin-bottom: 24px;"></div>
                                <p style="margin: 0 0 4px 0; font-size: 12px; color: #4B5563; line-height: 1.5;">If you didn't request this link, you can safely ignore this email.</p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px 48px 40px; text-align: center;">
                                <p style="margin: 0; font-size: 13px; font-weight: 500; color: #E84A3C;">Made with ❤️ for food lovers</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

        return self.send_email(to_email, subject, html_body, text_body)




    def send_access_request_confirmation(self, to_email: str) -> bool:
        subject = "TasteBud — Access Request Received"

        text_body = f"""
    Hi there!

    We've received your request to join TasteBud.

    Our team will review your request and you'll hear back soon.

    - The TasteBud Team
    """

        html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; background-color: #111111; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #111111;">
            <tr>
                <td align="center" style="padding: 20px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width: 480px; background-color: #111111;">
                        <tr>
                            <td style="padding: 56px 40px 20px 40px; text-align: center;">
                                <h1 style="margin: 0 0 8px 0; font-size: 38px; font-weight: 700; letter-spacing: -0.5px;">
                                    <span style="color: #E84A3C;">Taste</span><span style="color: #FF6B4A;">Bud</span>
                                </h1>
                                <div style="width: 48px; height: 3px; background: linear-gradient(90deg, #E84A3C, #FF6B4A); margin: 0 auto 40px auto; border-radius: 2px;"></div>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px; text-align: center;">
                                <h2 style="margin: 0 0 8px 0; font-size: 22px; font-weight: 700; color: #F5F5F5; letter-spacing: -0.3px;">Request Received</h2>
                                <p style="margin: 0 0 32px 0; font-size: 15px; color: #6B7280; line-height: 1.5;">We've received your request to join TasteBud. Our team will review it and you'll hear back soon.</p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 32px;">
                                <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                                    <tr>
                                        <td style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 24px 20px; text-align: center;">
                                            <p style="margin: 0; font-size: 15px; color: #9CA3AF; line-height: 1.6;">You'll receive another email once your access has been approved.</p>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 40px 40px 16px 40px; text-align: center;">
                                <div style="width: 100%; height: 1px; background: rgba(255, 255, 255, 0.06); margin-bottom: 24px;"></div>
                                <p style="margin: 0 0 4px 0; font-size: 12px; color: #4B5563; line-height: 1.5;">If you didn't request access, you can safely ignore this email.</p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px 48px 40px; text-align: center;">
                                <p style="margin: 0; font-size: 13px; font-weight: 500; color: #E84A3C;">Made with &#10084;&#65039; for food lovers</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

        return self.send_email(to_email, subject, html_body, text_body)

    def send_access_request_notification(self, requester_email: str, token: str) -> bool:
        approve_url = f"{settings.BACKEND_BASE_URL}/api/v1/early-access/approve?token={token}"
        deny_url = f"{settings.BACKEND_BASE_URL}/api/v1/early-access/deny?token={token}"
        subject = f"TasteBud Access Request: {requester_email}"

        text_body = f"""
    New TasteBud access request from: {requester_email}

    Approve: {approve_url}
    Deny: {deny_url}

    - TasteBud System
    """

        html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
    </head>
    <body style="margin: 0; padding: 0; background-color: #111111; font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
        <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background-color: #111111;">
            <tr>
                <td align="center" style="padding: 20px;">
                    <table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="max-width: 480px; background-color: #111111;">
                        <tr>
                            <td style="padding: 56px 40px 20px 40px; text-align: center;">
                                <h1 style="margin: 0 0 8px 0; font-size: 38px; font-weight: 700; letter-spacing: -0.5px;">
                                    <span style="color: #E84A3C;">Taste</span><span style="color: #FF6B4A;">Bud</span>
                                </h1>
                                <div style="width: 48px; height: 3px; background: linear-gradient(90deg, #E84A3C, #FF6B4A); margin: 0 auto 40px auto; border-radius: 2px;"></div>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px; text-align: center;">
                                <h2 style="margin: 0 0 8px 0; font-size: 22px; font-weight: 700; color: #F5F5F5; letter-spacing: -0.3px;">New Access Request</h2>
                                <p style="margin: 0 0 32px 0; font-size: 15px; color: #6B7280; line-height: 1.5;">Someone wants to join TasteBud</p>
                            </td>
                        </tr>
                        <!-- Requester email -->
                        <tr>
                            <td style="padding: 0 32px;">
                                <table role="presentation" width="100%" cellpadding="0" cellspacing="0">
                                    <tr>
                                        <td style="background: rgba(255, 255, 255, 0.04); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 16px; padding: 24px 20px; text-align: center;">
                                            <p style="margin: 0 0 4px 0; font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: 1px;">Requester</p>
                                            <p style="margin: 0; font-size: 18px; font-weight: 600; color: #F5F5F5;">{requester_email}</p>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        <!-- Action buttons -->
                        <tr>
                            <td style="padding: 32px 40px 0 40px; text-align: center;">
                                <table role="presentation" cellpadding="0" cellspacing="0" style="margin: 0 auto;">
                                    <tr>
                                        <td style="border-radius: 14px; background: linear-gradient(135deg, #22C55E 0%, #16A34A 100%); box-shadow: 0 8px 24px rgba(34, 197, 94, 0.25); padding: 0; margin-right: 12px;">
                                            <a href="{approve_url}" target="_blank" style="display: inline-block; padding: 16px 36px; font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; font-weight: 700; color: #FFFFFF; text-decoration: none; letter-spacing: -0.2px;">Approve</a>
                                        </td>
                                        <td style="width: 16px;"></td>
                                        <td style="border-radius: 14px; background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); box-shadow: 0 8px 24px rgba(239, 68, 68, 0.25); padding: 0;">
                                            <a href="{deny_url}" target="_blank" style="display: inline-block; padding: 16px 36px; font-family: 'Inter', -apple-system, sans-serif; font-size: 16px; font-weight: 700; color: #FFFFFF; text-decoration: none; letter-spacing: -0.2px;">Deny</a>
                                        </td>
                                    </tr>
                                </table>
                            </td>
                        </tr>
                        <!-- Footer -->
                        <tr>
                            <td style="padding: 40px 40px 16px 40px; text-align: center;">
                                <div style="width: 100%; height: 1px; background: rgba(255, 255, 255, 0.06); margin-bottom: 24px;"></div>
                                <p style="margin: 0 0 4px 0; font-size: 12px; color: #4B5563; line-height: 1.5;">These links are single-use. Clicking one will invalidate the other.</p>
                            </td>
                        </tr>
                        <tr>
                            <td style="padding: 0 40px 48px 40px; text-align: center;">
                                <p style="margin: 0; font-size: 13px; font-weight: 500; color: #E84A3C;">TasteBud Admin</p>
                            </td>
                        </tr>
                    </table>
                </td>
            </tr>
        </table>
    </body>
    </html>
    """

        return self.send_email(settings.EARLY_ACCESS_ADMIN_EMAIL, subject, html_body, text_body)


email_service = EmailService()
