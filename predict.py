"""
predict.py

Thin CLI for one-off local testing. Used to load the model directly and
duplicate main.py's prediction logic -- now it just POSTs to the running API,
so there's one place (main.py) that actually knows how to classify a ticket.

Run:
    uvicorn main:app --reload      # in another terminal
    python predict.py "My card still hasn't arrived"
"""

import sys

import httpx

API_URL = "http://127.0.0.1:8000/classify"


def main():
    if len(sys.argv) > 1:
        ticket_text = " ".join(sys.argv[1:])
    else:
        ticket_text = input("Enter a support ticket: ").strip()

    if not ticket_text:
        print("No ticket text provided.")
        return

    try:
        response = httpx.post(API_URL, json={"text": ticket_text}, timeout=10)
        response.raise_for_status()
    except httpx.ConnectError:
        print(f"Couldn't reach {API_URL} -- is `uvicorn main:app` running?")
        return
    except httpx.HTTPStatusError as exc:
        print(f"API returned an error: {exc.response.status_code} {exc.response.text}")
        return

    data = response.json()
    if data["needs_review"]:
        print(f"NEEDS REVIEW (top guess {data['top_k'][0]['category']!r} at only {data['confidence']:.0%} confidence)")
    else:
        print(f"Predicted category: {data['category']} ({data['confidence']:.0%} confidence)")
    print("Top candidates:")
    for item in data["top_k"]:
        print(f"  {item['category']}: {item['probability']:.0%}")


if __name__ == "__main__":
    main()
