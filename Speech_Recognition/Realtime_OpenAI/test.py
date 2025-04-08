import websockets
import asyncio
import sys

print(f"Python executable: {sys.executable}")
print(f"Using websockets version: {websockets.__version__}")
print(f"websockets module path: {websockets.__file__}")

async def test():
    async with websockets.connect(
        "wss://echo.websocket.org",
        extra_headers={"Custom-Header": "Test"}
    ) as ws:
        await ws.send("Hello")
        response = await ws.recv()
        print(f"Received: {response}")

if __name__ == "__main__":
    asyncio.run(test())