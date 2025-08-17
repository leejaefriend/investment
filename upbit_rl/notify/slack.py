import os, json, requests, time
from typing import Optional, List, Dict, Any

class SlackNotifier:
    """Send Slack messages via Incoming Webhook or Bot Token (chat.postMessage)."""
    def __init__(self, webhook_url: Optional[str]=None, bot_token: Optional[str]=None, channel: Optional[str]=None, timeout: int=10):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.channel = channel or os.getenv("SLACK_CHANNEL")
        self.timeout = timeout

    def send(self, text: str, blocks: Optional[List[Dict[str, Any]]]=None) -> bool:
        if self.bot_token:
            return self._send_web_api(text, blocks)
        if self.webhook_url:
            return self._send_webhook(text, blocks)
        print("[SlackNotifier] No webhook_url or bot_token set.")
        return False

    def _send_webhook(self, text: str, blocks: Optional[List[Dict[str, Any]]]):
        payload: Dict[str, Any] = {"text": text}
        if blocks: payload["blocks"] = blocks
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                return True
            print(f"[SlackNotifier] Webhook failed: {resp.status_code} {resp.text}")
        except Exception as e:
            print(f"[SlackNotifier] Webhook error: {e}")
        return False

    def _send_web_api(self, text: str, blocks: Optional[List[Dict[str, Any]]]):
        if not self.channel:
            print("[SlackNotifier] channel is required for Web API.")
            return False
        headers = {"Authorization": f"Bearer {self.bot_token}", "Content-Type": "application/json; charset=utf-8"}
        payload: Dict[str, Any] = {"channel": self.channel, "text": text}
        if blocks: payload["blocks"] = blocks
        try:
            resp = requests.post("https://slack.com/api/chat.postMessage", headers=headers, json=payload, timeout=self.timeout)
            data = resp.json()
            if data.get("ok"):
                return True
            print(f"[SlackNotifier] Web API failed: {data}")
        except Exception as e:
            print(f"[SlackNotifier] Web API error: {e}")
        return False

    def heartbeat(self, service_name: str="upbit-rl"):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        return self.send(f":heartbeat: {service_name} alive @ {ts}")