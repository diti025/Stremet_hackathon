import json
import uuid

users = {
    "employee": {
        "username": "employee1",
        "email": "employee@company.com",
        "password": "emp123",
        "token": str(uuid.uuid4()),
        "role": "employee"
    },
    "manager": {
        "username": "manager1",
        "email": "manager@company.com",
        "password": "mgr123",
        "token": str(uuid.uuid4()),
        "role": "manager"
    }
}

with open("users.json", "w") as f:
    json.dump(users, f, indent=4)

print("users.json created with demo employee and manager")