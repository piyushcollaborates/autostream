# tools.py - Lead capture tool for AutoStream agent

collected_lead = {
    "name": None,
    "email": None,
    "platform": None
}

def mock_lead_capture(name: str, email: str, platform: str):
    print("\n" + "="*50)
    print("LEAD CAPTURED SUCCESSFULLY!")
    print("="*50)
    print(f"Name     : {name}")
    print(f"Email    : {email}")
    print(f"Platform : {platform}")
    print("="*50 + "\n")
    return f"Lead captured successfully: {name}, {email}, {platform}"

def reset_lead():
    collected_lead["name"] = None
    collected_lead["email"] = None
    collected_lead["platform"] = None

def is_lead_complete():
    return all([
        collected_lead["name"],
        collected_lead["email"],
        collected_lead["platform"]
    ])