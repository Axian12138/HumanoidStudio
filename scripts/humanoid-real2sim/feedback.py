"""Search action/task examples and record user corrections for RSS real2sim."""

from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
EXAMPLES = REPOSITORY / "examples/humanoid-real2sim/cases"
CASES = EXAMPLES / "feedback"
STAGES = ("scene", "retarget", "track", "visualization", "archive", "cross-stage", "repository")
CATEGORIES = ("input", "geometry", "interaction", "motion", "physics", "tooling", "delivery", "data", "code", "test")
STATUSES = ("open", "fixed", "verified")
KINDS = ("error", "new_requirement", "new_problem")
ROUTE_TARGETS = ("skill", "tool", "example")
ROUTE_DECISIONS = ("update", "none", "defer")


def load_cases() -> list[dict]:
    return [json.loads(path.read_text()) for path in sorted(CASES.glob("*.json"))]


def load_examples() -> list[dict]:
    path = EXAMPLES / "index.json"
    return json.loads(path.read_text()) if path.exists() else []


def save_case(case: dict) -> Path:
    path = CASES / f"{case['id']}.json"
    path.write_text(json.dumps(case, ensure_ascii=False, indent=2) + "\n")
    return path


def valid_id(value: str) -> bool:
    return re.fullmatch(r"[a-z0-9][a-z0-9-]*", value) is not None


def parse_routes(values: list[str]) -> dict[str, dict[str, str]]:
    """Parse target=decision: reason, preserving explicit triage rationale."""
    routes = {}
    for value in values:
        target, separator, choice = value.partition("=")
        decision, reason_separator, reason = choice.partition(":")
        if (
            not separator
            or not reason_separator
            or target not in ROUTE_TARGETS
            or decision not in ROUTE_DECISIONS
            or not reason.strip()
        ):
            raise ValueError(f"route must be target=update|none|defer: reason: {value}")
        if target in routes:
            raise ValueError(f"duplicate route target: {target}")
        routes[target] = {"decision": decision, "reason": reason.strip()}
    return routes


def ready_to_verify(case: dict) -> bool:
    """Legacy records remain readable; new records need complete triage."""
    if not case.get("validation") or not case.get("fix"):
        return False
    if case.get("kind") == "error" and not case.get("cause"):
        return False
    if "routing" not in case:
        return True
    return all(case["routing"].get(target, {}).get("decision") in ROUTE_DECISIONS for target in ROUTE_TARGETS)


def tokens(value: str) -> set[str]:
    english = re.findall(r"[a-z0-9_]+", value.casefold())
    chinese = re.findall(r"[\u3400-\u9fff]+", value)
    return set(english + [part[i : i + 2] for part in chinese for i in range(len(part) - 1)])


def score(query: str, text: str) -> int:
    if not query:
        return 1
    text = text.casefold()
    query = query.casefold()
    result = 10 if query in text else 0
    result += len(tokens(query) & tokens(text))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add", help="record a user correction in natural language")
    add.add_argument("--id", required=True)
    add.add_argument("--stage", choices=STAGES, required=True)
    add.add_argument("--category", choices=CATEGORIES, required=True)
    add.add_argument("--impact", choices=("high", "medium", "low"), required=True)
    add.add_argument("--kind", choices=KINDS, required=True)
    for field in ("task", "actions", "title", "narrative", "expected", "observed", "source"):
        add.add_argument(f"--{field}", required=True)
    add.add_argument("--example-id", default="")
    add.add_argument("--cause", default="")
    add.add_argument("--route", action="append", default=[], help="target=update|none|defer: reason")
    update = sub.add_parser("update", help="add a fix, validation, or linked example")
    update.add_argument("id")
    for field in ("fix", "validation", "source", "narrative", "example_id", "cause"):
        update.add_argument(f"--{field}")
    update.add_argument("--kind", choices=KINDS)
    update.add_argument("--route", action="append", default=[], help="target=update|none|defer: reason")
    update.add_argument("--status", choices=STATUSES)
    update.add_argument("--recurrences", type=int)
    search = sub.add_parser("search", help="rank language-rich examples and feedback together")
    search.add_argument("--query", default="")
    search.add_argument("--task", default="")
    search.add_argument("--action", default="")
    search.add_argument("--stage", choices=STAGES)
    search.add_argument("--category", choices=CATEGORIES)
    search.add_argument("--kind", choices=("all", "examples", "feedback"), default="all")
    show = sub.add_parser("show", help="read an example narrative or a feedback record")
    show.add_argument("id")
    args = parser.parse_args()

    if args.command == "add":
        if not valid_id(args.id):
            parser.error("id must use lowercase letters, digits and hyphens")
        if (CASES / f"{args.id}.json").exists():
            parser.error(f"case already exists: {args.id}")
        examples = {item["id"] for item in load_examples()}
        if args.example_id and args.example_id not in examples:
            parser.error(f"unknown example: {args.example_id}")
        case = {
            key: getattr(args, key)
            for key in (
                "id",
                "kind",
                "stage",
                "category",
                "impact",
                "task",
                "title",
                "narrative",
                "expected",
                "observed",
                "source",
                "example_id",
            )
        }
        case["actions"] = [part.strip() for part in args.actions.split(",") if part.strip()]
        if not case["actions"]:
            parser.error("actions must contain at least one action")
        try:
            routes = parse_routes(args.route)
        except ValueError as exc:
            parser.error(str(exc))
        case.update(
            date=date.today().isoformat(),
            status="open",
            recurrences=1,
            cause=args.cause,
            routing=routes,
            fix="",
            validation="",
        )
        print(save_case(case))
    elif args.command == "update":
        if not valid_id(args.id):
            parser.error("id must use lowercase letters, digits and hyphens")
        path = CASES / f"{args.id}.json"
        if not path.exists():
            parser.error(f"unknown case: {args.id}")
        case = json.loads(path.read_text())
        for field in (
            "fix",
            "validation",
            "source",
            "narrative",
            "example_id",
            "cause",
            "kind",
            "status",
            "recurrences",
        ):
            value = getattr(args, field)
            if value is not None:
                case[field] = value
        try:
            routes = parse_routes(args.route)
        except ValueError as exc:
            parser.error(str(exc))
        if routes:
            case.setdefault("routing", {}).update(routes)
        if case["recurrences"] < 1:
            parser.error("recurrences must be positive")
        if case["status"] == "verified" and not ready_to_verify(case):
            parser.error("verified requires a fix, validation, error cause, and skill/tool/example routing decisions")
        print(save_case(case))
    elif args.command == "show":
        for item in load_examples():
            if item["id"] == args.id:
                print((EXAMPLES / item["file"]).read_text())
                return
        path = CASES / f"{args.id}.json"
        if not path.exists():
            parser.error(f"unknown example or feedback: {args.id}")
        print(path.read_text())
    else:
        results = []
        if args.kind in ("all", "examples"):
            for item in load_examples():
                if args.task and not score(args.task, item["task"]):
                    continue
                if args.action and not score(args.action, " ".join(item["actions"] + item["keywords"])):
                    continue
                narrative = (EXAMPLES / item["file"]).read_text()
                rank = score(
                    args.query, " ".join((item["title"], item["summary"], " ".join(item["keywords"]), narrative))
                )
                if rank:
                    results.append(
                        (
                            rank,
                            f"example {item['id']} [{item['task']}/{item['status']}] "
                            f"{item['title']} — {item['summary']} ({EXAMPLES / item['file']})",
                        )
                    )
        if args.kind in ("all", "feedback"):
            for item in load_cases():
                if args.stage and item["stage"] != args.stage:
                    continue
                if args.category and item["category"] != args.category:
                    continue
                if args.task and not score(args.task, item.get("task", "")):
                    continue
                if args.action and not score(args.action, " ".join(item.get("actions", []))):
                    continue
                rank = score(args.query, json.dumps(item, ensure_ascii=False))
                if rank:
                    results.append(
                        (
                            rank,
                            f"feedback {item['id']} [{item['stage']}/{item['status']}] "
                            f"{item['title']} ({CASES / (item['id'] + '.json')})",
                        )
                    )
        for _, line in sorted(results, key=lambda result: (-result[0], result[1])):
            print(line)


if __name__ == "__main__":
    main()
