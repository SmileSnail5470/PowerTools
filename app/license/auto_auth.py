import json
import hashlib
import logging
import os
import requests 
import pathlib
import secrets
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import app.library._machine_id as machine_id
from huggingface_hub import hf_hub_download, HfApi

logger = logging.getLogger("License")

CENT = Decimal("0.01")


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def money(value: Decimal) -> Decimal:
    return Decimal(value).quantize(CENT, rounding=ROUND_HALF_UP)


def money_text(value: Decimal) -> str:
    return f"{money(value):.2f}"

@dataclass(frozen=True)
class PricingTier:
    key: str
    name: str
    price_text: str
    unit_text: str
    popular: bool = False


@dataclass(frozen=True)
class Quote:
    days: int
    tier_key: str
    amount: Decimal

    @property
    def amount_text(self) -> str:
        return money_text(self.amount)


class PricingPolicy:
    YEAR_DAYS = 365
    YEAR_PRICE = Decimal("150.00")
    PRICE_BASIC = Decimal("1.00")
    PRICE_SHORT = Decimal("0.60")
    PRICE_LONG = Decimal("0.50")
    MIN_DAYS = 1
    MAX_DAYS = 3650

    TIERS: Tuple[PricingTier, ...] = (
        PricingTier("basic", "基础日常 (<30天)", "¥1.00", "/天"),
        PricingTier("short", "短期优惠 (1-3个月)", "¥0.60", "/天"),
        PricingTier("long", "长期优惠 (>3个月)", "¥0.50", "/天"),
        PricingTier("year", "年度订阅 (365天)", "¥150.00", "/年", True),
    )

    @classmethod
    def clamp_days(cls, days: Any) -> int:
        try:
            days = int(days)
        except (TypeError, ValueError):
            days = cls.MIN_DAYS
        return max(cls.MIN_DAYS, min(cls.MAX_DAYS, days))

    @classmethod
    def quote(cls, days: Any) -> Quote:
        days = cls.clamp_days(days)
        if days >= cls.YEAR_DAYS:
            years, rest = divmod(days, cls.YEAR_DAYS)
            amount = cls.YEAR_PRICE * years + cls.PRICE_LONG * rest
            tier_key = "year"
        elif days > 90:
            amount = cls.PRICE_LONG * days
            tier_key = "long"
        elif days >= 30:
            amount = cls.PRICE_SHORT * days
            tier_key = "short"
        else:
            amount = cls.PRICE_BASIC * days
            tier_key = "basic"
        return Quote(days=days, tier_key=tier_key, amount=money(amount))


class OrderStatus(str, Enum):
    CREATED = "created"                  # 已创建，尚未展示收款码
    WAITING_PAYMENT = "waiting_payment"  # 已展示收款码，等待到账
    USER_CLAIMED_PAID = "user_claimed"   # 用户自称已付款，等待核对
    PAID = "paid"                        # 远端确认到账
    ISSUING = "issuing"                  # 远端正在签发
    ISSUED = "issued"                    # 已拿到 .lic 文件
    ACTIVATED = "activated"              # 已在本机激活
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass
class AuthOrder:
    order_id: str
    machine_id: str
    days: int
    tier_key: str
    amount: str
    payable_amount: str
    created_at: str
    status: str = OrderStatus.CREATED.value
    channel: str = "qrcode"
    paid_at: str = ""
    issued_at: str = ""
    activated_at: str = ""
    transaction_id: str = ""
    license_id: str = ""
    license_path: str = ""
    message: str = ""
    app_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AuthOrder":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def amount_text(self) -> str:
        return self.payable_amount or self.amount

    @property
    def is_closed(self) -> bool:
        return self.status in (
            OrderStatus.ACTIVATED.value,
            OrderStatus.CANCELLED.value,
            OrderStatus.TIMEOUT.value,
        )


def new_order_id(machine_fingerprint: str = "") -> str:
    stamp = datetime.now().strftime("%y%m%d")
    prefix = (machine_fingerprint or "0000")[-4:].upper()
    alphabet = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    rand = "".join(secrets.choice(alphabet) for _ in range(4))
    return f"PT{stamp}-{prefix}-{rand}"


class AuthOrderStore:
    INDEX_NAME = "orders.json"
    TRAIL_NAME = "orders_audit.jsonl"
    GENESIS_HASH = "0" * 64

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = os.path.join(pathlib.Path.home(), ".PowerTools", "license")
        self._base_dir = base_dir
        os.makedirs(self._base_dir, exist_ok=True)
        self._index_file = os.path.join(self._base_dir, self.INDEX_NAME)
        self._trail_file = os.path.join(self._base_dir, self.TRAIL_NAME)
        self._lock = threading.RLock()

    @property
    def base_dir(self) -> str:
        return self._base_dir

    @property
    def trail_file(self) -> str:
        return self._trail_file

    def _load_index(self) -> Dict[str, Any]:
        if not os.path.exists(self._index_file):
            return {}
        try:
            with open(self._index_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Auth order index unreadable, ignored: {e}")
            return {}

    def _write_index(self, index: Dict[str, Any]):
        tmp = self._index_file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self._index_file)

    def save(self, order: AuthOrder, event: str, **detail: Any) -> AuthOrder:
        with self._lock:
            index = self._load_index()
            index[order.order_id] = order.to_dict()
            self._write_index(index)
            self._append_trail(order, event, detail)
        return order

    def get(self, order_id: str) -> Optional[AuthOrder]:
        data = self._load_index().get(order_id)
        return AuthOrder.from_dict(data) if data else None

    def list_orders(self) -> List[AuthOrder]:
        orders = [AuthOrder.from_dict(v) for v in self._load_index().values()]
        return sorted(orders, key=lambda o: o.created_at, reverse=True)

    def find_open_order(self) -> Optional[AuthOrder]:
        for order in self.list_orders():
            if not order.is_closed and order.status != OrderStatus.FAILED.value:
                return order
        return None

    def delete(self, order_id: str, reason: str = "user_deleted") -> bool:
        with self._lock:
            index = self._load_index()
            data = index.pop(order_id, None)
            if data is None:
                return False
            self._write_index(index)
            order = AuthOrder.from_dict(data)
            self._append_trail(order, "order_deleted", {"reason": reason})
        logger.info(f"Auth order record deleted: {order_id} reason={reason}")
        return True

    def delete_many(self, order_ids: List[str], reason: str = "user_deleted") -> int:
        return sum(1 for order_id in list(order_ids) if self.delete(order_id, reason=reason))

    def _last_trail_record(self) -> Optional[Dict[str, Any]]:
        if not os.path.exists(self._trail_file):
            return None
        last = None
        try:
            with open(self._trail_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        last = json.loads(line)
        except Exception as e:
            logger.warning(f"Auth audit trail unreadable: {e}")
            return None
        return last

    @staticmethod
    def _record_hash(record: Dict[str, Any]) -> str:
        payload = {k: v for k, v in record.items() if k != "hash"}
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _append_trail(self, order: AuthOrder, event: str, detail: Dict[str, Any]):
        last = self._last_trail_record()
        record = {
            "seq": (last.get("seq", 0) + 1) if last else 1,
            "ts": _now_iso(),
            "order_id": order.order_id,
            "machine_id": order.machine_id,
            "event": event,
            "status": order.status,
            "days": order.days,
            "amount": order.payable_amount or order.amount,
            "detail": detail or {},
            "prev_hash": last.get("hash", self.GENESIS_HASH) if last else self.GENESIS_HASH,
        }
        record["hash"] = self._record_hash(record)
        try:
            with open(self._trail_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to append auth audit trail: {e}")

    def read_trail(self, order_id: Optional[str] = None) -> List[Dict[str, Any]]:
        if not os.path.exists(self._trail_file):
            return []
        records = []
        with open(self._trail_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if order_id is None or record.get("order_id") == order_id:
                    records.append(record)
        return records

    def verify_trail(self) -> Tuple[bool, Optional[int]]:
        prev_hash = self.GENESIS_HASH
        for record in self.read_trail():
            if record.get("prev_hash") != prev_hash:
                return False, record.get("seq")
            if record.get("hash") != self._record_hash(record):
                return False, record.get("seq")
            prev_hash = record["hash"]
        return True, None


class PaymentStatus(str, Enum):
    PENDING = "pending"          # 未查到到账
    PAID = "paid"                # 已确认到账
    FAILED = "failed"            # 支付失败 / 金额不符
    EXPIRED = "expired"          # 订单超时关闭
    UNAVAILABLE = "unavailable"  # 校验通道不可用（未接入 / 网络异常）


@dataclass
class PaymentResult:
    status: PaymentStatus
    transaction_id: str = ""
    paid_at: str = ""
    message: str = ""


@dataclass
class IssueResult:
    available: bool = False
    license_text: str = ""
    license_id: str = ""
    message: str = ""


class PaymentVerifier(ABC):
    """支付到账校验接口。"""

    @abstractmethod
    def query(self, order: AuthOrder) -> PaymentResult:
        raise NotImplementedError


class LicenseIssuer(ABC):
    """许可证下发接口。"""

    @abstractmethod
    def fetch(self, order: AuthOrder) -> IssueResult:
        raise NotImplementedError


class DefaultPaymentVerifier(PaymentVerifier):
    """支付校验默认实现：无自动验证通道，返回 UNAVAILABLE, 让 poll() 直接尝试授权文件下发。"""
    REASON = "支付校验通道未接入，请联系作者手动核对"

    def query(self, order: AuthOrder) -> PaymentResult:
        return PaymentResult(status=PaymentStatus.UNAVAILABLE, message=self.REASON)


class DefaultLicenseIssuer(LicenseIssuer):
    HF_REPO_ID = "SmailSnail/PowerToolsLicense"
    HF_REPO_TYPE = "model"
    HF_TIMEOUT = 30
    REASON = "授权文件下发通道未接入，请联系作者手动签发"

    def fetch(self, order: AuthOrder) -> IssueResult:
        path_in_repo = f"{order.machine_id}/{order.order_id}.lic"
        try:
            api = HfApi()
            try:
                info = api.repo_info(repo_id=self.HF_REPO_ID, repo_type=self.HF_REPO_TYPE)
                existing_paths = {sibling.rfilename for sibling in (info.siblings or [])}
            except Exception as e:
                logger.warning(f"[HF] Failed to list repo files: {e}")
                return IssueResult(available=False, message=f"无法访问授权仓库：{e}")
            if path_in_repo not in existing_paths:
                return IssueResult(available=False, message="授权文件尚未生成，请稍候（作者核对后将上传）…")
            local_path = hf_hub_download(
                repo_id=self.HF_REPO_ID,
                filename=path_in_repo,
                repo_type=self.HF_REPO_TYPE,
            )
            with open(local_path, "r", encoding="utf-8") as f:
                license_text = f.read()
            if not license_text.strip():
                return IssueResult(available=False, message="授权文件内容为空，请联系作者")
            logger.info(f"[HF] License fetched successfully for {order.order_id}")
            return IssueResult(
                available=True,
                license_text=license_text,
                license_id=order.order_id,
                message="授权文件获取成功",
            )
        except Exception as e:
            logger.warning(f"[HF] Unexpected error fetching license for {order.order_id}: {e}")
            return IssueResult(available=False, message=f"获取授权文件时发生错误：{e}")


class WeChatWorkNotifier:
    WECHAT_WEBHOOK_URL = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=cf0d6142-5388-4cee-8b82-a72698775518"
    WECHAT_TIMEOUT = 10

    def __init__(self, webhook_url: str = WECHAT_WEBHOOK_URL):
        self._webhook_url = webhook_url

    def notify_order(self, order, extra: str = "", on_result: "Optional[callable]" = None) -> None:
        message = self._build_message(order, extra)
        thread = threading.Thread(
            target=self._send,
            args=(message, on_result),
            daemon=True,
            name=f"wechat-notify-{order.order_id}",
        )
        thread.start()

    def _build_message(self, order, extra: str) -> str:
        status_map = {
            "user_claimed": "💰 用户声明已付款，等待核对",
            "waiting_payment": "⏳ 等待到账",
            "paid": "✅ 已确认到账",
            "issued": "📄 授权文件已签发",
            "activated": "🎉 已在本机激活",
            "cancelled": "❌ 已取消",
        }
        status_text = status_map.get(order.status, order.status)
        lines = [
            "## 🔔 PowerTools 授权订单通知",
            "",
            f"> **订单号**: `{order.order_id}`",
            f"> **设备标识码**: `{order.machine_id}`",
            f"> **授权天数**: {order.days} 天",
            f"> **支付金额**: ¥{order.amount_text}",
            f"> **下单时间**: {order.created_at}",
            f"> **当前状态**: {status_text}",
        ]
        if extra:
            lines.append(f"> **备注**: {extra}")
        return "\n".join(lines)

    def _send(self, content: str, on_result: "Optional[callable]") -> None:
        success = False
        error = ""
        try:
            payload = {
                "msgtype": "markdown",
                "markdown": {"content": content},
            }
            resp = requests.post(self._webhook_url, json=payload, timeout=self.WECHAT_TIMEOUT)
            if resp.status_code == 200:
                result = resp.json()
                if result.get("errcode") == 0:
                    logger.info("[WeChat] Notification sent successfully")
                    success = True
                else:
                    error = f"API 返回错误: errcode={result.get('errcode')} errmsg={result.get('errmsg')}"
                    logger.warning(f"[WeChat] {error}")
            else:
                error = f"HTTP {resp.status_code}"
                logger.warning(f"[WeChat] {error}: {resp.text}")
        except Exception as e:
            error = str(e)
            logger.warning(f"[WeChat] Failed to send notification: {e}")
        if on_result is not None:
            try:
                on_result(success, error)
            except Exception as e:
                logger.warning(f"[WeChat] on_result callback raised: {e}")


class AuthStage(str, Enum):
    WAITING = "waiting"          # 等待支付
    PAID = "paid"                # 到账成功，等待签发
    ISSUED = "issued"            # 已下发并激活
    UNAVAILABLE = "unavailable"  # 通道未接入 / 暂时不可用
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class AuthProgress:
    stage: AuthStage
    message: str = ""
    license_path: str = ""
    order: Optional[AuthOrder] = None


class AutoAuthService:
    ORDER_TTL_SECONDS = 10 * 60

    def __init__(
        self,
        license_manager=None,
        store: Optional[AuthOrderStore] = None,
        verifier: Optional[PaymentVerifier] = None,
        issuer: Optional[LicenseIssuer] = None,
        policy=PricingPolicy,
        app_version: str = os.environ["POWERTOOLS_VERSION"],
    ):
        self._license_manager = license_manager
        self._store = store or AuthOrderStore()
        self._verifier = verifier or DefaultPaymentVerifier()
        self._issuer = issuer or DefaultLicenseIssuer()
        self._notifier = WeChatWorkNotifier()
        self._policy = policy
        self._app_version = app_version

    @property
    def store(self) -> AuthOrderStore:
        return self._store

    @property
    def tiers(self) -> Tuple[PricingTier, ...]:
        return self._policy.TIERS

    def quote(self, days: Any) -> Quote:
        return self._policy.quote(days)

    def list_orders(self) -> List[AuthOrder]:
        return self._store.list_orders()

    def delete_orders(self, order_ids: List[str], reason: str = "user_deleted") -> int:
        return self._store.delete_many(order_ids, reason=reason)

    def create_order(self, days: Any, channel: str = "qrcode") -> AuthOrder:
        quote = self.quote(days)
        fingerprint = ""
        try:
            fingerprint = machine_id.get_machine_id()
        except Exception as e:
            logger.warning(f"Failed to read machine id when creating order: {e}")
        order = AuthOrder(
            order_id=new_order_id(fingerprint),
            machine_id=fingerprint,
            days=quote.days,
            tier_key=quote.tier_key,
            amount=quote.amount_text,
            payable_amount=quote.amount_text,
            created_at=_now_iso(),
            status=OrderStatus.CREATED.value,
            channel=channel,
            app_version=self._app_version,
        )
        self._store.save(order, event="order_created")
        logger.info(f"Auth order created: {order.order_id} days={order.days} amount={order.amount_text}")
        return order

    def send_order(self, order: AuthOrder, on_failure=None):
        def _on_result(success: bool, error: str):
            if not success:
                logger.warning(f"Order notification failed for {order.order_id}: {error}")
                if on_failure is not None:
                    on_failure(error)
        try:
            self._notifier.notify_order(
                order,
                extra="用户声明已完成支付，请核对到账情况并签发授权文件",
                on_result=_on_result,
            )
            logger.info(f"Order notification dispatched: {order.order_id}")
        except Exception as e:
            logger.warning(f"Failed to dispatch order notification: {e}")
            if on_failure is not None:
                on_failure(str(e))

    def mark_waiting_payment(self, order: AuthOrder) -> AuthOrder:
        order.status = OrderStatus.WAITING_PAYMENT.value
        return self._store.save(order, event="payment_qr_shown", channel=order.channel)

    def mark_user_claimed_paid(self, order: AuthOrder, note: str = "") -> AuthOrder:
        order.status = OrderStatus.USER_CLAIMED_PAID.value
        return self._store.save(order, event="user_claimed_paid", note=note)

    def cancel(self, order: AuthOrder, reason: str = "user_cancelled") -> AuthOrder:
        if order.status in (OrderStatus.ACTIVATED.value, OrderStatus.ISSUED.value):
            return order
        order.status = OrderStatus.CANCELLED.value
        order.message = reason
        return self._store.save(order, event="order_cancelled", reason=reason)

    def is_expired(self, order: AuthOrder) -> bool:
        try:
            created = datetime.fromisoformat(order.created_at.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            return False
        elapsed = (datetime.now(timezone.utc) - created).total_seconds()
        return elapsed > self.ORDER_TTL_SECONDS

    def poll(self, order: AuthOrder) -> AuthProgress:
        if order.status == OrderStatus.ACTIVATED.value:
            return AuthProgress(AuthStage.ISSUED, "许可证已激活", order.license_path, order)

        if self.is_expired(order):
            order.status = OrderStatus.TIMEOUT.value
            self._store.save(order, event="order_timeout")
            return AuthProgress(AuthStage.TIMEOUT, "订单已超时关闭，请重新下单", order=order)

        if order.status not in (OrderStatus.PAID.value, OrderStatus.ISSUING.value, OrderStatus.ISSUED.value):
            result = self._query_payment(order)
            if result.status == PaymentStatus.UNAVAILABLE:
                # 支付校验通道不可用（人工核对场景）：
                # 跳过支付验证，直接尝试拉取授权文件。
                # 若 LicenseIssuer 查到文件说明作者已核对并签发，视为付款确认。
                order.status = OrderStatus.PAID.value
                order.paid_at = _now_iso()
                order.transaction_id = result.transaction_id
                self._store.save(order, event="payment_confirmed", transaction_id=result.transaction_id)
                return self._issue_and_activate(order)
            if result.status == PaymentStatus.PENDING:
                return AuthProgress(AuthStage.WAITING, result.message or "等待自动下发文件...", order=order)
            if result.status in (PaymentStatus.FAILED, PaymentStatus.EXPIRED):
                order.status = (OrderStatus.FAILED if result.status == PaymentStatus.FAILED else OrderStatus.TIMEOUT).value
                order.message = result.message
                self._store.save(order, event="payment_failed", reason=result.message)
                stage = AuthStage.FAILED if result.status == PaymentStatus.FAILED else AuthStage.TIMEOUT
                return AuthProgress(stage, result.message or "支付未完成", order=order)
            order.status = OrderStatus.PAID.value
            order.paid_at = result.paid_at or _now_iso()
            order.transaction_id = result.transaction_id
            self._store.save(order, event="payment_confirmed", transaction_id=result.transaction_id)

        return self._issue_and_activate(order)

    def _query_payment(self, order: AuthOrder) -> PaymentResult:
        try:
            return self._verifier.query(order)
        except NotImplementedError:
            return PaymentResult(PaymentStatus.UNAVAILABLE, message=DefaultPaymentVerifier.REASON)
        except Exception as e:
            logger.warning(f"Payment query failed for {order.order_id}: {e}")
            return PaymentResult(PaymentStatus.UNAVAILABLE, message=f"到账校验暂时不可用：{e}")

    def _issue_and_activate(self, order: AuthOrder) -> AuthProgress:
        if order.status == OrderStatus.PAID.value:
            order.status = OrderStatus.ISSUING.value
            self._store.save(order, event="issue_requested")
        try:
            issue = self._issuer.fetch(order)
        except NotImplementedError:
            return AuthProgress(AuthStage.UNAVAILABLE, DefaultLicenseIssuer.REASON, order=order)
        except Exception as e:
            logger.warning(f"License fetch failed for {order.order_id}: {e}")
            return AuthProgress(AuthStage.UNAVAILABLE, f"许可证下发暂时不可用：{e}", order=order)
        if not issue.available or not issue.license_text:
            return AuthProgress(
                AuthStage.PAID,
                issue.message or "支付已确认，许可证正在签发，请稍候...",
                order=order,
            )
        try:
            path = self.deliver_license(order, issue.license_text, license_id=issue.license_id)
        except Exception as e:
            order.status = OrderStatus.FAILED.value
            order.message = str(e)
            self._store.save(order, event="activation_failed", reason=str(e))
            return AuthProgress(AuthStage.FAILED, f"许可证激活失败：{e}", order=order)
        return AuthProgress(AuthStage.ISSUED, "支付成功！授权文件已生成", path, order)

    def license_inbox_dir(self) -> str:
        path = os.path.join(self._store.base_dir, "issued")
        os.makedirs(path, exist_ok=True)
        return path

    def deliver_license(self, order: AuthOrder, license_text: str, license_id: str = "") -> str:
        path = os.path.join(self.license_inbox_dir(), f"{order.order_id}.lic")
        with open(path, "w", encoding="utf-8") as f:
            f.write(license_text)
        order.status = OrderStatus.ISSUED.value
        order.issued_at = _now_iso()
        order.license_path = path
        order.license_id = license_id
        self._store.save(
            order,
            event="license_issued",
            license_id=license_id,
            sha256=hashlib.sha256(license_text.encode("utf-8")).hexdigest(),
        )

        if self._license_manager is not None:
            self._license_manager.activate(path)
            order.status = OrderStatus.ACTIVATED.value
            order.activated_at = _now_iso()
            data = getattr(self._license_manager, "license_data", None)
            if data is not None and getattr(data, "license_id", ""):
                order.license_id = data.license_id
            self._store.save(order, event="license_activated", license_id=order.license_id)
        return path

    def support_summary(self, order: AuthOrder) -> str:
        return (
            f"订单号：{order.order_id}\n"
            f"设备标识码：{order.machine_id}\n"
            f"授权天数：{order.days} 天\n"
            f"支付金额：¥{order.amount_text}\n"
            f"下单时间：{order.created_at}\n"
            f"当前状态：{order.status}"
        )
