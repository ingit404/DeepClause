import os
import imaplib
import email
from bs4 import BeautifulSoup
from crewai_tools import tool
from agentic_approach.config import EMAIL_USER, EMAIL_PASSWORD

@tool("search_emails")
def search_emails(query: str) -> dict:
    """
    Search recent email threads by keyword.
    Use ONLY for approvals, rejections, follow-ups, or recent discussions.
    """
    mail = imaplib.IMAP4_SSL("imap.gmail.com")
    mail.login(EMAIL_USER, EMAIL_PASSWORD)

    # Inbox only; change to "[Gmail]/All Mail" if needed
    mail.select("inbox")

    _, data = mail.search(None, f'(TEXT "{query}")')
    email_ids = data[0].split()[-5:]  # last 5 messages

    results = []

    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/html":
                    body = BeautifulSoup(
                        part.get_payload(decode=True),
                        "html.parser"
                    ).get_text()

        results.append({
            "from": msg.get("From"),
            "subject": msg.get("Subject"),
            "date": msg.get("Date"),
            "snippet": body[:300]
        })

    return {"emails": results}
