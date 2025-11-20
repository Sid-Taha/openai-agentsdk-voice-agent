# openai-agent-sdk/main.py
import asyncio
import numpy as np
import sounddevice as sd
from dotenv import load_dotenv
from agents.realtime import RealtimeAgent, RealtimeRunner
from agents import set_tracing_export_api_key
from collections import deque
import threading
from agents.mcp import MCPServerStreamableHttp, MCPServerStreamableHttpParams

load_dotenv()

# Get API key from environment variable
import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
set_tracing_export_api_key(OPENAI_API_KEY)

# Audio configuration
SAMPLE_RATE = 24000
CHANNELS = 1
INPUT_CHUNK_SIZE = 1024
OUTPUT_CHUNK_SIZE = 1024

# Better audio buffer using deque
class AudioBuffer:
    def __init__(self):
        self.buffer = deque()
        self.lock = threading.Lock()
    
    def write(self, data):
        """Add audio data to buffer"""
        with self.lock:
            # Convert bytes to numpy array
            audio_array = np.frombuffer(data, dtype=np.int16)
            # Add each sample to the buffer
            for sample in audio_array:
                self.buffer.append(sample)
    
    def read(self, num_samples):
        """Read audio samples from buffer"""
        with self.lock:
            if len(self.buffer) >= num_samples:
                # Get the requested number of samples
                samples = [self.buffer.popleft() for _ in range(num_samples)]
                return np.array(samples, dtype=np.int16)
            else:
                # Not enough data, return what we have padded with zeros
                samples = [self.buffer.popleft() for _ in range(len(self.buffer))]
                samples.extend([0] * (num_samples - len(samples)))
                return np.array(samples, dtype=np.int16)
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()

# Create buffers
input_buffer = deque()
output_buffer = AudioBuffer()

def audio_input_callback(indata, frames, time, status):
    """Callback for microphone input"""
    if status:
        print(f"Input status: {status}")
    audio_data = (indata * 32767).astype(np.int16)
    input_buffer.append(audio_data.tobytes())

def audio_output_callback(outdata, frames, time, status):
    """Callback for speaker output"""
    if status:
        print(f"Output status: {status}")
    
    try:
        # Read samples from buffer
        samples = output_buffer.read(frames)
        # Convert to float32 and normalize
        outdata[:, 0] = samples.astype(np.float32) / 32768.0
    except Exception as e:
        print(f"Playback error: {e}")
        outdata.fill(0)

async def main():
    print("🎤 Initializing Voice Agent with MCP...")
    print("🔌 Connecting to MCP Server at http://127.0.0.1:9000/mcp")
    
    # MCP Server connection setup
    connection = MCPServerStreamableHttpParams(url="http://127.0.0.1:9000/mcp")
    
    async with MCPServerStreamableHttp(params=connection) as server:
        print("✅ MCP Server connected!")
        
        agent = RealtimeAgent(
            name="HealthcareAgent",
            instructions=(
                "You are a helpful medical receptionist assistant. "
                "Your goal is to check claim status for clients. "
                "1. First, greet the user and ask for their registered Phone Number. "
                "2. Once you get the number, use the 'fetch_claim_status' tool immediately. "
                "3. Tell the user the details returned by the tool in a conversational way. "
                "Speak clearly and politely."
            ),
            mcp_servers=[server]  # MCP server pass karo
        )

        runner = RealtimeRunner(
            starting_agent=agent,
            config={
                "model_settings": {
                    "model_name": "gpt-realtime-mini",
                    "voice": "alloy",
                    "modalities": ["audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16",
                    "input_audio_transcription": {"model": "gpt-4o-mini-transcribe"},
                    "turn_detection": {"type": "semantic_vad", "interrupt_response": True},
                }
            },
        )

        session = await runner.run()

        # Start audio streams
        input_stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=audio_input_callback,
            blocksize=INPUT_CHUNK_SIZE
        )
        
        output_stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype=np.float32,
            callback=audio_output_callback,
            blocksize=OUTPUT_CHUNK_SIZE
        )

        with input_stream, output_stream:
            print("\n✅ Voice Agent Ready! (Connected to MCP)")
            print("🎤 Speak into your microphone")
            print("💬 Try saying: 'Hi, check claim status for number 12345'")
            print("🛑 Press Ctrl+C to stop\n")
            
            async with session:
                # Task to send audio from microphone to session
                async def send_audio():
                    while True:
                        try:
                            if input_buffer:
                                audio_data = input_buffer.popleft()
                                await session.send_audio(audio_data)
                            else:
                                await asyncio.sleep(0.01)
                        except Exception as e:
                            print(f"Send error: {e}")
                            break

                send_task = asyncio.create_task(send_audio())

                try:
                    async for event in session:
                        if event.type == "tool_start":
                            print(f"\n🔧 Calling MCP tool: {event.tool.name}")
                        
                        elif event.type == "tool_end":
                            print(f"✅ MCP Tool result: {event.output}\n")

                        elif event.type == "audio":
                            try:
                                if hasattr(event.audio, 'data'):
                                    audio_bytes = event.audio.data
                                else:
                                    audio_bytes = event.audio
                                
                                if isinstance(audio_bytes, bytes) and len(audio_bytes) > 0:
                                    # Write to buffer immediately
                                    output_buffer.write(audio_bytes)
                            except Exception as e:
                                print(f"Audio error: {e}")

                        elif event.type == "audio_end":
                            print("✅ Agent finished speaking\n")

                        elif event.type == "error":
                            print(f"\n❌ ERROR: {event.error}")
                            break

                except Exception as e:
                    print(f"\n❌ Session error: {e}")
                finally:
                    send_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Voice session ended. Goodbye!")