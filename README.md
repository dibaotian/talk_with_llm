# talk with llm

#### Install
##### Server
- #>git clone https://github.com/dibaotian/talk_with_llm.git
- #>cd talk_with_llm/server
- #>python -m venv .venv
- #>source .venv/bin/activate
- #>python server.py

##### Client
- copy the client.py to target machine
- python client.py --host <host_ip>

#### Topology 
##### client(PCM 16000 Int16 singal channel)<---udp--->server(PCM 24000 Int16 singal channel)

#### Server pipline
##### UDP PCM receiver channel->VAD(google model)->STT(sensevoice/wishper)->LLM(Qwen2 7B)->TTS(Chattts)->UDP PCM send channel

#### Feature
- UDP connection. client connect with server through UDP （send 9527/receive 2795）
- stream mode process.the client gather PCM(Int16) stream form micphone and directly send to Server
- supporting long-duration voice interactions. VAD make the system is capable of continuous speech recognition and processing without relying on a wake word
- multi language support, the STT use the wisper_lagre_v3 or cosevoise(alibaba)
- LLM use Qwen2_7b
- TTS use ChatTTS， support Chinese and english

#### limitation
- only one connection supported, will move to webrtc
- While system is processing current conversiation, it cannot be immediately interrupted to handle a new one."





### Other
##### the server was tested at python 3.10
##### you may install ffmpeg in the system