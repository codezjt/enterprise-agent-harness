import requests, json

BASE = "http://127.0.0.1:8000"

def post(path, body):
    r = requests.post(BASE + path, json=body)
    print(f"  [{r.status_code}] {json.dumps(r.json(), ensure_ascii=False)}")
    return r

print("=== Step 1: viewer (no file:write) -> 403 ===")
post("/v1/tools/execute", {
    "tool_name": "write_file",
    "arguments": {"path": "D:/Users/00796472/Desktop/test.txt", "content": "0000"},
    "principal_id": "u1",
    "principal_role": "viewer",
})

print()
print("=== Step 2: editor (HIGH risk) -> REQUIRE_APPROVAL ===")
r = post("/v1/tools/execute", {
    "tool_name": "write_file",
    "arguments": {"path": "D:/Users/00796472/Desktop/test.txt", "content": "0000"},
    "principal_id": "u2",
    "principal_role": "editor",
})
approval_id = r.json().get("approval_id")

print()
print(f"=== Step 3: Approve ({approval_id}) ===")
post("/v1/tools/approval/approve", {
    "approval_id": approval_id,
    "comment": "approved by manager",
})

print()
print("=== Step 4: Retry with approval_id -> SUCCESS ===")
post("/v1/tools/execute", {
    "tool_name": "write_file",
    "arguments": {"path": "D:/Users/00796472/Desktop/test.txt", "content": "0000"},
    "principal_id": "u2",
    "principal_role": "editor",
    "approval_id": approval_id,
})

print()
print("=== Step 5: Verify by reading back ===")
post("/v1/tools/execute", {
    "tool_name": "read_file",
    "arguments": {"path": "D:/Users/00796472/Desktop/test.txt"},
    "principal_id": "u1",
    "principal_role": "viewer",
})
