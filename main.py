from fastapi import FastAPI

app = FastAPI()

# Define your routes and other FastAPI configurations here


def run():
    print(f"start server with fastapi dev api.py")
    # import uvicorn

    # uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    run()
