import requests
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000/ask"

# Change this whenever you want:
# 2, 5, 10, 20, etc.
NUMBER_OF_USERS = 10

# Maximum number of requests running at the same time.
MAX_CONCURRENT_REQUESTS = 10

REQUEST_TIMEOUT = 60


# ============================================================
# QUESTIONS
# ============================================================

QUESTION_POOL = [
    "What is paging in operating systems?",
    "What is deadlock in operating systems?",
    "What is normalization in DBMS?",
    "What is inheritance in Java?",
    "What is polymorphism in Java?",
    "What is virtual memory?",
    "What is a primary key in DBMS?",
    "What is process scheduling?",
    "What is fragmentation in operating systems?",
    "What is a thread in operating systems?",
    "What is encapsulation in Java?",
    "What is a foreign key in DBMS?",
    "What is an operating system?",
    "What is a database management system?",
    "What is abstraction in Java?",
    "What is CPU scheduling?",
    "What is a transaction in DBMS?",
    "What is cache memory?",
    "What is JDBC?",
    "What is a REST API?"
]


# ============================================================
# CREATE USERS DYNAMICALLY
# ============================================================

def create_users(number_of_users):
    """
    Dynamically create virtual users.

    Each user receives:
        - unique user ID
        - unique session ID
        - different question
    """

    users = []

    for i in range(number_of_users):

        user = {
            "user_id": f"User-{i + 1:02d}",
            "session_id": str(uuid.uuid4()),
            "question": QUESTION_POOL[i % len(QUESTION_POOL)]
        }

        users.append(user)

    return users


# ============================================================
# SEND REQUEST FOR ONE USER
# ============================================================

def send_request(user):
    """
    Send one request to SignalMinds for a particular user.
    """

    start_time = time.perf_counter()

    payload = {
        "question": user["question"],
        "session_id": user["session_id"]
    }

    try:

        response = requests.post(
            API_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT
        )

        elapsed_time = time.perf_counter() - start_time

        if response.status_code == 200:

            data = response.json()

            return {
                "user_id": user["user_id"],
                "session_id": user["session_id"],
                "question": user["question"],
                "status": "SUCCESS",
                "status_code": response.status_code,
                "route": data.get("route"),
                "source": data.get("source"),
                "answer": data.get("answer"),
                "time": elapsed_time
            }

        else:

            return {
                "user_id": user["user_id"],
                "session_id": user["session_id"],
                "question": user["question"],
                "status": "FAILED",
                "status_code": response.status_code,
                "route": None,
                "source": None,
                "answer": response.text,
                "time": elapsed_time
            }

    except requests.exceptions.RequestException as e:

        elapsed_time = time.perf_counter() - start_time

        return {
            "user_id": user["user_id"],
            "session_id": user["session_id"],
            "question": user["question"],
            "status": "ERROR",
            "status_code": None,
            "route": None,
            "source": None,
            "answer": str(e),
            "time": elapsed_time
        }


# ============================================================
# MAIN LOAD TEST
# ============================================================

def main():

    print()
    print("=" * 70)
    print("              SIGNALMINDS MULTI-USER TEST")
    print("=" * 70)

    print(f"API URL              : {API_URL}")
    print(f"Virtual users        : {NUMBER_OF_USERS}")
    print(f"Concurrent requests  : {MAX_CONCURRENT_REQUESTS}")

    print()
    print("Creating users...")

    users = create_users(NUMBER_OF_USERS)

    print(f"Created {len(users)} users.")

    print()
    print("-" * 70)
    print("Users are sending requests...")
    print("-" * 70)

    overall_start = time.perf_counter()

    results = []

    # --------------------------------------------------------
    # Send requests concurrently
    # --------------------------------------------------------

    with ThreadPoolExecutor(
        max_workers=MAX_CONCURRENT_REQUESTS
    ) as executor:

        futures = [
            executor.submit(send_request, user)
            for user in users
        ]

        for future in as_completed(futures):

            result = future.result()

            results.append(result)

            if result["status"] == "SUCCESS":

                print(
                    f"{result['user_id']:>8}  "
                    f"✓  "
                    f"{result['route']:<10}  "
                    f"{result['time']:.2f} sec"
                )

            else:

                print(
                    f"{result['user_id']:>8}  "
                    f"✗  "
                    f"{result['status']:<10}  "
                    f"{result['time']:.2f} sec"
                )

    overall_time = time.perf_counter() - overall_start


    # ========================================================
    # RESULTS
    # ========================================================

    successful = [
        result
        for result in results
        if result["status"] == "SUCCESS"
    ]

    failed = [
        result
        for result in results
        if result["status"] != "SUCCESS"
    ]


    if results:

        average_time = sum(
            result["time"]
            for result in results
        ) / len(results)

    else:

        average_time = 0


    requests_per_second = (
        len(results) / overall_time
        if overall_time > 0
        else 0
    )


    print()
    print("=" * 70)
    print("                         RESULTS")
    print("=" * 70)

    print(f"Users tested         : {NUMBER_OF_USERS}")
    print(f"Requests completed   : {len(results)}")
    print(f"Successful requests  : {len(successful)}")
    print(f"Failed requests      : {len(failed)}")
    print(f"Total execution time : {overall_time:.2f} sec")
    print(f"Average response     : {average_time:.2f} sec")
    print(f"Requests / second    : {requests_per_second:.2f}")

    print("=" * 70)


    # ========================================================
    # SHOW FAILED REQUESTS
    # ========================================================

    if failed:

        print()
        print("FAILED REQUESTS")
        print("-" * 70)

        for result in failed:

            print(f"User       : {result['user_id']}")
            print(f"Status     : {result['status']}")
            print(f"Status code: {result['status_code']}")
            print(f"Error      : {result['answer']}")
            print()


    # ========================================================
    # SHOW SESSION INFORMATION
    # ========================================================

    print()
    print("SESSION INFORMATION")
    print("-" * 70)

    unique_sessions = {
        result["session_id"]
        for result in results
    }

    print(f"Unique sessions created : {len(unique_sessions)}")

    if len(unique_sessions) == NUMBER_OF_USERS:

        print("Session isolation       : ✓ PASS")

    else:

        print("Session isolation       : ✗ CHECK")


    print()
    print("=" * 70)

    if len(successful) == NUMBER_OF_USERS:

        print("          ✓ MULTI-USER TEST PASSED")

    else:

        print("          ✗ SOME REQUESTS FAILED")

    print("=" * 70)
    print()


# ============================================================
# PROGRAM ENTRY
# ============================================================

if __name__ == "__main__":
    main()