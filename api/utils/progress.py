"""进度计算与本地文件状态检查"""
from __future__ import annotations

from pathlib import Path

_EXPECTED_REPORTS = ["需求分析.json", "PRD.json", "报价单.json"]


def _calculate_progress(status: str) -> int:
    progress_map = {
        "queued": 0,
        "analyzing": 15,
        "designing": 30,
        "estimating": 45,
        "completed": 50,
        "pending_development": 50,
        "developing": 75,
        "development_completed": 100,
        "development_failed": 50,
        "failed": -1,
    }
    return progress_map.get(status, 0)


def _check_local_status(user_id: str, ticket_id: str, project_root: Path) -> dict:
    """检查工单在本地文件系统中的实际状态。"""
    ticket_dir = project_root / "data" / "users" / user_id / ticket_id
    result = {
        "directory_exists": False,
        "ticket_json_exists": False,
        "has_report": False,
        "report_files": [],
        "expected_reports": [],
        "missing_reports": [],
        "report_status": "not_expected",
        "has_product": False,
        "product_file_count": 0,
        "product_sample": [],
        "local_deleted": False,
        "is_empty_workspace": True,
    }

    if not ticket_dir.exists():
        result["local_deleted"] = True
        return result

    result["directory_exists"] = True

    ticket_json = ticket_dir / "工单" / "工单.json"
    if ticket_json.exists():
        result["ticket_json_exists"] = True
        result["is_empty_workspace"] = False

    report_dir = ticket_dir / "报告"
    if report_dir.exists():
        try:
            report_files = []
            for rpt in _EXPECTED_REPORTS:
                if (report_dir / rpt).is_file():
                    report_files.append(rpt)
            if report_files:
                result["has_report"] = True
                result["report_files"] = sorted(report_files)
                result["is_empty_workspace"] = False
        except OSError:
            pass

    product_dir = ticket_dir / "成品"
    if product_dir.exists():
        try:
            all_files = []
            for f in product_dir.rglob("*"):
                if f.is_file() and not any(p.startswith(".") for p in f.parts):
                    all_files.append(str(f.relative_to(product_dir)))
            if all_files:
                result["has_product"] = True
                result["product_file_count"] = len(all_files)
                result["product_sample"] = sorted(all_files)[:5]
                result["is_empty_workspace"] = False
        except OSError:
            pass

    return result


def _enrich_local_status(ticket: dict, local_status: dict) -> dict:
    """根据数据库中的数据，与本地文件系统状态做对比。"""
    expected = []
    if ticket.get("analysis"):
        expected.append("需求分析.json")
    if ticket.get("prd"):
        expected.append("PRD.json")
    if ticket.get("quote"):
        expected.append("报价单.json")

    local_status["expected_reports"] = expected
    actual = set(local_status["report_files"])
    missing = [r for r in expected if r not in actual]

    if not expected:
        local_status["report_status"] = "not_expected"
    elif not missing:
        local_status["report_status"] = "complete"
    elif actual:
        local_status["report_status"] = "partial"
    else:
        local_status["report_status"] = "missing"

    local_status["missing_reports"] = missing
    return local_status
