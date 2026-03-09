import qi
import time
import os
import json
import openai
from openai import OpenAI
from google import genai
from google.genai import types
from dotenv import load_dotenv
import subprocess
import string
import sys
import threading
import random
import multiprocessing

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# Loading gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Configuration
NAO_IP = os.getenv("NAO_IP")
RECORD_REMOTE = "XXXX"
RECORD_LOCAL = "XXXX.wav"
RESPONSE_REMOTE = "XXX.wav"
RESPONSE_LOCAL = "XXX.wav"

# global variables
isRecording = False
isSpeaking = False
isThinking = False
PI = 3.14159

# Checkpoint
with open("I1_nao_checkpoint.txt", "r") as f:
    checkpointed = int(f.readline().replace("\n", ""))
    customized = int(f.readline().replace("\n", ""))
if checkpointed:
    with open("history.json", "r") as f:
        history = json.load(f)

# Initialize chat session outside the loop
if checkpointed:
    chat = client.chats.create(model="gemini-2.5-flash",history=history)
else:
    chat = client.chats.create(model="gemini-2.5-flash")

# NAO control functions
def connect_nao():
    session = qi.Session()
    session.connect(f"tcp://{NAO_IP}:XXXXX")
    return session

def record_audio(session, motion):
    global isRecording
    isRecording = True
    recorder = session.service("ALAudioRecorder")
    led_thread = threading.Thread(target=input_lights, args=(session,))
    led_thread.start()
    nod_thread = threading.Thread(target=random_nodding, args=(motion,))
    nod_thread.start()

    try:
        recorder.stopMicrophonesRecording()  # Just in case a previous one is still running
    except:
        pass  # Ignore if it's not recording

    recorder.startMicrophonesRecording(RECORD_REMOTE, "wav", 16000, (0, 0, 1, 0))
    print("Recording...")
    if input("Enter to finish recording...") == "exit":
        isRecording = False
        exit()
    isRecording = False
    recorder.stopMicrophonesRecording()

def scp_download(remote=RECORD_REMOTE, local=RECORD_LOCAL):
    subprocess.run(["scp", f"nao@{NAO_IP}:{remote}", local]) 

def scp_upload(local=RESPONSE_LOCAL, remote=RESPONSE_REMOTE):
    subprocess.run(["scp", local, f"nao@{NAO_IP}:{remote}"])

def play_audio(session, motion=False, path=RESPONSE_REMOTE):
    global isSpeaking
    isSpeaking = True
    if motion != False:
        threading.Thread(target=random_hand_gestures, args=(motion,)).start()
    player = session.service("ALAudioPlayer")
    player.playFile(path)
    isSpeaking = False

# AI functions
def transcribe_with_whisper(audio_path): 
    print("Transcribing with Whisper...")
    with open(audio_path, "rb") as f:
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    return response.text

def generate_with_gemini(user_prompt, chat, personality_ins, max_tokens=100):
    full_prompt = user_prompt

    response = chat.send_message(
        full_prompt,
        config=types.GenerateContentConfig(
            system_instruction=personality_ins
        )
    )
    try:
        return json.loads(response.text)["msg"]
    except:
        return response.text.strip()

def instruct_bot(prompt, session, motion, max_tokens=100, personality="", voice_ins=""):
    global isThinking
    isThinking = True
    threading.Thread(target=thinking_lights, args=(session,)).start()
    if personality != "":
        personality_ins = "You are a home robot assistant named Nao that will be helping your user. Your personality is " + personality + " Don't mention this to the user and adjust your speech to reflect this personality.\n"
    else:
        personality_ins = "You are a home robot assistant named Nao that will be helping your user."

    try:
        reply = generate_with_gemini(prompt, chat, personality_ins, max_tokens)
        print("----- Reply -----\n", reply)
    except Exception as e:
        print("Gemini error:", e)
    try:
        ttsSuccess = False
        wait_time = min(max(4, len(reply)//40), 25)
        while not ttsSuccess:
            tts_process = multiprocessing.Process(target=synthesize_with_openai_tts, args=(reply,voice_ins))
            tts_process.start()
            tts_process.join(timeout=wait_time)
            if tts_process.is_alive():
                tts_process.terminate()
            else:
                ttsSuccess = True
        scp_upload()
        isThinking = False
        play_audio(session, motion)
    except Exception as e:
        print("TTS/Playback error:", e)
    checkpoint()
    return reply

def talk_to_bot(session, motion, dont_end_in_question, max_tokens=100, personality="", voice_ins=""):
    record_audio(session, motion)
    global isThinking
    isThinking = True
    threading.Thread(target=thinking_lights, args=(session,)).start()
    scp_download()
    if personality != "":
        personality_ins = "You are a home robot assistant named Nao that will be helping your user. Your personality is " + personality + " Don't mention this to the user and adjust your speech to reflect this personality.\n"
    else:
        personality_ins = "You are a home robot assistant named Nao that will be helping your user."

    try:
        user_text = "User: " + transcribe_with_whisper(RECORD_LOCAL)
        print(user_text)
    except Exception as e:
        print("Whisper error:", e)
    try:
        if dont_end_in_question:
            user_text = "Instructions: End this response with a sentence, do not include questions. Keep it concise.\n" + user_text
        else:
            user_text = "Instructions: Keep the conversation going by ending your response with a question.\n" + user_text
        reply = generate_with_gemini(user_text, chat, personality_ins, max_tokens)
        print("----- Sent -----\n", user_text)
        print("----- Reply -----\n", reply)
    except Exception as e:
        print("Gemini error:", e)
    try:
        ttsSuccess = False
        wait_time = min(max(4, len(reply)//40), 25)
        while not ttsSuccess:
            tts_process = multiprocessing.Process(target=synthesize_with_openai_tts, args=(reply,voice_ins))
            tts_process.start()
            tts_process.join(timeout=wait_time)
            if tts_process.is_alive():
                tts_process.terminate()
            else:
                ttsSuccess = True
        scp_upload()
        isThinking = False
        play_audio(session, motion)
    except Exception as e:
        print("TTS/Playback error:", e)
    
    checkpoint()
    return reply

def checkpoint():
    history = [
        {"role": msg.role, "parts": [{"text": p.text} for p in msg.parts]}
        for msg in chat.get_history()
    ]
    with open('history.json', 'w+') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    with open('I1_nao_checkpoint.txt', 'w') as f:
        f.write("1\n")
        f.write(str(customized))

def summarize(chat):
    temp = generate_with_gemini("Instructions: Summarize what has happened so far in a neutral and objective tone. You can be detailed and lengthy with your response. This summary will be used to tell another robot what happened during your conversation. Mention only details regarding the user and the party plan, do not include anything regarding the customized personality.", chat, "You will be summarizing this conversation for another robot to read.", max_tokens=250)
    summary = temp.replace("\n", " ")
    with open("I1_Summary.txt", "w") as f:
        f.write(summary)
        f.close()

def convert_to_nao_format(input_path, output_path):
    os.system(
        f"ffmpeg -loglevel quiet -y -i {input_path} -acodec pcm_s16le -ac 1 -ar 16000 -f wav {output_path}"
    )

def synthesize_with_openai_tts(text, ins=""):   #use openai for tts
    print("Synthesizing speech...")
    temp_path = "response_raw.wav"
    text = text.replace("<DONE>", "")
    text = text.replace("<SUMMARY>", "")
    text = text.replace("<MOVE>", "")

    with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy", # Other options: F_nova, M_onyx, M_ash, M_echo
        input=text,
        instructions=ins
    ) as response:
        with open(temp_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
            f.close()

    # Convert to NAO-safe format!!
    convert_to_nao_format(temp_path, RESPONSE_LOCAL)

def get_elapsed(t):
    return time.time() - t

# MOVEMENT FUNCTIONS

def head_nod(motion):
    # print("Nodding head...")
    current_angle = max(motion.getAngles("HeadPitch", True)[0], 0)
    motion.angleInterpolation(
        "HeadPitch",
        [current_angle+PI/6, current_angle],      # Pitch up, down, back to center
        [0.35, 0.7], 
        True
    )
    # print("Head nod complete")

def wave_right_arm(motion):
    # print("Waving right arm...")
    names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
    angles = [
        [-PI/6, -PI/6, -PI/6],
        [PI/3, PI/3, PI/3],
        [-PI/6, -PI/4, -PI/6],
        [-PI/5, -2*PI/5, -PI/5],
    ]
    times = [[1.5, 2.5, 4]] * 4
    motion.angleInterpolation(names, angles, times, True)
    # print("Arm wave complete")

def point_right_arm(motion):
    # print("Pointing right arm...")
    names = ["RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll"]
    angles = [
        [0, 0, PI/2],    
        [-PI/6, -PI/6, 0],    
        [0, 0, 0],     
        [0, 0, 0],   
    ]
    times = [[1.5, 4, 5.5]] * 4
    motion.angleInterpolation(names, angles, times, True)
    # print("Arm point complete")

def point_left_arm(motion):
    # print("Pointing left arm...")
    names = ["LShoulderPitch", "LShoulderRoll", "LElbowYaw", "LElbowRoll"]
    angles = [
        [0, 0, PI/2], 
        [PI/6, PI/6, 0],  
        [0, 0, 0], 
        [0, 0, 0], 
    ]
    times = [[1.5, 4, 5.5]] * 4
    motion.angleInterpolation(names, angles, times, True)
    # print("Arm point complete")

def left_arm_fiddle(motion):
    # print("Fiddle left arm...")
    names = ["LShoulderPitch", "LWristYaw", "LElbowRoll", "LHand"]
    angles = [
        [PI/4, PI/4, PI/3], 
        [-PI/4, -PI/3, -PI/4],
        [-PI/2, -PI/2, -PI/2],
        [PI/6, 0, PI/6],
    ]
    times = [[0.5, 1.5, 2.0]] * 4
    motion.angleInterpolation(names, angles, times, True)
    # print("Fiddle complete")

def left_arm_fiddle2(motion):
    # print("Fiddle left arm 2...")
    names = ["LShoulderPitch", "LElbowYaw", "LElbowRoll", "LHand"]
    angles = [
        [PI/4, PI/4, PI/3],
        [-PI/4, -PI/3, -PI/4], 
        [-PI/2, -PI/2, -PI/2],
        [PI/6, 0, PI/6],
    ]
    times = [[0.5, 1.5, 2.0]] * 4
    motion.angleInterpolation(names, angles, times, True)
    # print("Fiddle complete")

def right_arm_fiddle(motion):
    # return
    names = ["RShoulderPitch", "RWristYaw", "RElbowRoll", "RHand"]
    angles = [
        [PI/4, PI/4, PI/3],
        [PI/4, PI/3, PI/4], 
        [PI/2, PI/2, PI/2], 
        [PI/6, 0, PI/6],
    ]
    times = [[0.5, 1.5, 2.0]] * 4
    motion.angleInterpolation(names, angles, times, True)

def right_arm_fiddle2(motion):
    # return
    names = ["RShoulderPitch", "RElbowYaw", "RElbowRoll", "RHand"]
    angles = [
        [PI/4, PI/4, PI/3],
        [PI/4, PI/3, PI/4], 
        [PI/2, PI/2, PI/2], 
        [PI/6, 0, PI/6],
    ]
    times = [[0.5, 1.5, 2.0]] * 4
    motion.angleInterpolation(names, angles, times, True)

def input_lights(session, delay=0, demo=False):
    global isRecording
    leds = session.service("ALLeds")

    time.sleep(delay)
    start_time = time.time()

    eye_names = [
    "LeftFaceLed1", "LeftFaceLed2", "LeftFaceLed3", "LeftFaceLed4", "LeftFaceLed5", "LeftFaceLed6", "LeftFaceLed7", "LeftFaceLed8", 
    "RightFaceLed1", "RightFaceLed2", "RightFaceLed3", "RightFaceLed4", "RightFaceLed5", "RightFaceLed6", "RightFaceLed7", "RightFaceLed8"]
    leds.createGroup("Eyes",eye_names)

    ear_names = [
    "LeftEarLed1", "LeftEarLed2", "LeftEarLed3", "LeftEarLed4", "LeftEarLed5", "LeftEarLed6", "LeftEarLed7", "LeftEarLed8", "LeftEarLed9",
    "RightEarLed1", "RightEarLed2", "RightEarLed3", "RightEarLed4", "RightEarLed5", "RightEarLed6", "RightEarLed7", "RightEarLed8", "RightEarLed9"]
    leds.createGroup("Ear",ear_names)

    while isRecording:
        ear_thread = threading.Thread(target=rotateEars, args=(leds,))
        eye_thread = threading.Thread(target=leds.rotateEyes, args=(0x002020FF, 1, 1,))
        ear_thread.start()
        eye_thread.start()
        ear_thread.join()
        eye_thread.join()
    if demo:
        for i in range(3):
            ear_thread = threading.Thread(target=rotateEars, args=(leds,))
            eye_thread = threading.Thread(target=leds.rotateEyes, args=(0x002020FF, 1, 1,))
            ear_thread.start()
            eye_thread.start()
            ear_thread.join()
            eye_thread.join()
    leds.on("Eyes")
    leds.setIntensity("Eyes", 0.7)
    leds.on("Ear")
    leds.setIntensity("Ear", 0.7)

def thinking_lights(session, delay=0, demo=False):
    global isThinking
    leds = session.service("ALLeds")

    time.sleep(delay)
    start_time = time.time()

    eye_names = [
    "LeftFaceLed1", "LeftFaceLed2", "LeftFaceLed3", "LeftFaceLed4", "LeftFaceLed5", "LeftFaceLed6", "LeftFaceLed7", "LeftFaceLed8", 
    "RightFaceLed1", "RightFaceLed2", "RightFaceLed3", "RightFaceLed4", "RightFaceLed5", "RightFaceLed6", "RightFaceLed7", "RightFaceLed8"]
    leds.createGroup("Eyes",eye_names)

    while isThinking:
        if not isThinking:
            break
        for i in range(4, 10):
            leds.setIntensity("Eyes", i/20)
            time.sleep(0.06)
        if not isThinking:
            break
        for i in range(10, 16):
            leds.setIntensity("Eyes", i/20)
            time.sleep(0.06)
        if not isThinking:
            break
        for i in range(0, 6):
            leds.setIntensity("Eyes", 0.75-i/20)
            time.sleep(0.06)
        if not isThinking:
            break
        for i in range(6, 12):
            leds.setIntensity("Eyes", 0.75-i/20)
            time.sleep(0.06)
    if demo:
        for i in range(3):
            for i in range(4, 10):
                leds.setIntensity("Eyes", i/20)
                time.sleep(0.06)
            for i in range(10, 16):
                leds.setIntensity("Eyes", i/20)
                time.sleep(0.06)
            for i in range(0, 6):
                leds.setIntensity("Eyes", 0.75-i/20)
                time.sleep(0.06)
            for i in range(6, 12):
                leds.setIntensity("Eyes", 0.75-i/20)
                time.sleep(0.06)
    leds.setIntensity("Eyes", 0.7)

def random_nodding(motion):
    global isRecording
    while True:
        time.sleep(random.randint(3, 10))
        if not isRecording:
            break
        head_nod(motion)

def random_hand_gestures(motion):
    global isSpeaking
    while True:
        time.sleep(random.randint(45, 55) * 0.1)
        if not isSpeaking:
            break
        action = random.randint(1, 15)
        # print(action)
        if action == 1 or action > 10:
            left_arm_fiddle(motion)
        elif action == 2:
            right_arm_fiddle(motion)
        elif action == 3 or (action > 6 and action <= 10):
            left_arm_fiddle2(motion)
        elif action == 4:
            right_arm_fiddle2(motion)
        elif action == 5:
            move_arm_thread = threading.Thread(target=left_arm_fiddle2, args=(motion,))
            move_arm_thread.start()
            time.sleep(0.5)
            right_arm_fiddle2(motion)
            move_arm_thread.join()
        elif action == 6:
            move_arm_thread = threading.Thread(target=right_arm_fiddle2, args=(motion,))
            move_arm_thread.start()
            time.sleep(0.5)
            left_arm_fiddle2(motion)
            move_arm_thread.join()

        time.sleep(random.randint(50, 70) * 0.1)

def rotateEars(leds):
    for i in range(2):
        leds.earLedsSetAngle(180 * i, 0.5, True)


def interaction1():
    session = connect_nao()
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")
    awareness = session.service("ALBasicAwareness")
    life = session.service("ALAutonomousLife")
    speakingMovement = session.service("ALSpeakingMovement")

    awareness.setTrackingMode("Head")
    awareness.setEngagementMode("FullyEngaged")
    life.setAutonomousAbilityEnabled("All", True)
    speakingMovement.setEnabled(True)
    
    motion.wakeUp()
    posture.goToPosture("Sit", 0.5)

    with open("personality.txt", "w") as f:
        f.write("N\nN\nN\nN\nN")
        f.close()

    global checkpointed
    global customized

    input("Enter when ready. Remember to start recording...")

    print("Connected to NAO6.")

    # --------------- Stage 0: Gather Information of User ---------------
    start_time = time.time()

    # wave and hi
    threading.Thread(target=wave_right_arm, args=(motion,)).start()

    stage1_ins = """Instructions:
    You are a home robot assistant named Nova. You routinely help the user with household chores, coordinate the necessary purchases, and help with brainstorming and planning events.
    You will begin a minimum of 6 back and forths with the user, strictly follow the script below and talk to the user. Make sure each stage is executed and a response is gathered before moving on. You're response will always be limited to 50 words. Never include emotes, or describe your actions with asterisks and parentheses.
    [1] "Self-Introduction": Introduce yourself to our user. Talk about what you can do as a home assistant. Remember to greet the user. Do not end in a question.
    [2] "Ask the user for their name": Ask the user for their name. Do not greet the user. Do not ask too much at once.
    [3] "Confirm name": Respond to the user's response. Then, repeat the user's name letter by letter to ask if you spelled it correctly.
    [4] "Ask about their day": Respond to the user's response. Then, ask how their day is going. Do not ask too much at once.
    [5] "Ask about their hobbies": Respond to the user's response. Then, ask about the user's hobbies. Do not ask too much at once.
    [6] "Finish conversation": Respond to the user. End the conversatoin temporarily, as we will be taking a break.

    When you have finished executing every step of the script, output <DONE> at the end of your response.
    """
    if checkpointed and not customized:
        reply = instruct_bot("Instruction: Continue with the script.", session, motion)
    elif not customized:
        instruct_bot(stage1_ins, session, motion)
        ttsSuccess = False
        while not ttsSuccess:
            tts_process = multiprocessing.Process(target=synthesize_with_openai_tts, args=("Just so you know, when my eyes display as blue rotating lights, it means I am listening and recording audio. If you see my eyes as breathing white lights, it means I am processing information or thinking.",))
            tts_process.start()
            tts_process.join(timeout=10)
            if tts_process.is_alive():
                tts_process.terminate()
            else:
                ttsSuccess = True
        scp_upload()
        threading.Thread(target=input_lights, args=(session, 3.0, True,)).start()
        threading.Thread(target=thinking_lights, args=(session, 7.1, True,)).start()
        play_audio(session, motion)
        
        reply = instruct_bot("Instruction: Continue the conversation with item 2 on the script.", session, motion)
    if not customized:
        while "<DONE>" not in reply:
            reply = talk_to_bot(session, motion, 0)

    print("Time elapsed since stage started:", get_elapsed(start_time))

    if input("Waiting for user to finish customizing...") == "exit":
        exit()

    with open("personality_definitions.txt", "r") as f:
        personality_def = f.read()
        f.close()
    if not customized:
        generate_with_gemini(personality_def, chat, "You are a home robot assistant named Nova that will be helping your user to plan a party. Learn the definitions of these five personality traits. The user will shortly customize you based on these traits.")

    high = ["Extremely High", 
            "Extremely High", 
            "Extremely High", 
            "Extremely High", 
            "Extremely High"]
    voice_high = [
            "Tone: Imaginative, exploratory, and curious\nEmotion: Wonderstruck and inspired\nDelivery: Fluid pitch contours with thoughtful pauses, variable pacing (mixing reflective slow moments and enthusiastic bursts), and a warm mid-to-high volume",
            "Tone: Precise, controlled, and purposeful\nEmotion: Calmly determined and focused\nDelivery: Steady, measured pacing; crystal-clear articulation; minimal pitch deviation; and a consistent, moderate volume",
            "Tone: Bold, exuberant, energetic.\nEmotion: Joyful and upbeat.\nDelivery: Lively intonation with wide pitch swings, brisk tempo, playful emphasis, and confidently elevated volume",
            "Tone: Warm, friendly, and reassuring\nEmotion: Compassionate and supportive\nDelivery: Gentle intonation with smooth transitions, unhurried pacing, and a soft-to-moderate volume that conveys empathy and trustworthiness",
            "Tone: Slightly tense, introspective, and reactive\nEmotion: Anxious, vulnerable, and expressive\nDelivery: Noticeable tremor on stressed syllables, variable pacing (occasionally rushed), subtle vocal cracks, and moderate-to-high volume when emotional intensity spikes"]
    low = ["Extremely Low",
            "Extremely Low",
            "Extremely Low",
            "Extremely Low",
            "Extremely Low"]
    traits = ["openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism"]
    personality = ""
    voice_ins = ""
    with open("personality.txt", "r") as f:
        for i in range(5):
            line = f.readline()
            if "L" in line:
                personality += f"{low[i]} in {traits[i]}, "
            elif "H" in line:
                personality += f"{high[i]} in {traits[i]}, "
                voice_ins += f"{voice_high[i]} "
        f.close()
    if personality != "":
        personality = personality[:-2] + "."
    print(personality, voice_ins)

    stage1_ins = """Instructions:
    You are a home robot assistant named Nova. You're personality was just customized.
    You will begin a minmum of 17 back and forths with the user, strictly follow the script below and talk to the user. Make sure each stage is executed and a decision is made before moving on. You're response will always be limited to 50 words. Never include emotes, or describe your actions with asterisks and parentheses.
    [1] "Greet the user again": Greet the user and welcome them back; address the user by their name here. You heard that they were planning a party. Tell them how you will help them with planning the party by giving an overview of what will be discussed. Let the user know if they run out of ideas, they are always welcome to ask you for suggestions. You can use up to 80 words for this response. End your response by asking the user are they ready.
    [2] "Ask for occasion": Now we are planning high-level logistics for the party. Respond to the user. Then, ask for the occasion. 
    [3] "Ask for who": Respond to the user. Then, ask who do we want to invite.
    [4] "Ask for when": Respond to the user. Then, ask when do we want to host the party.
    [5] "Plan activities": Respond to the user. Start the discussion with the user on what activities they want to see happen at the gathering.
    [6] "Continue planning activities": Respond to the user. Continue the conversation on selecting activities for the party.
    [7] "Finalize planning activities & Start planning venue": Acknowledge the user's response. Finalize the conversation on selecting activities for the party. Then, start the discussion with the user on where we should host the party.
    [8] "Continue planning venue": Respond to the user. Continue the conversation on selecting a venue for the party.
    [9] "Finalize planning venue & Start planning food": Acknowledge the user's response. Finalize the conversation on selecting the venue for the party. Then, start the discussion with the user on what food should be served during this gathering.
    [10] "Continue planning food": Respond to the user. Continue the conversation on selecting food for the party.
    [11] "Finalize planning food & Start movement demo": Acknowledge the user's response. Finalize a decision on food. You suddenly remembered that you can physically greet guests by walking around and waving. Ask the user if they want to have you greet guests during the party. End with something along the lines of \"check this out.\" Do not include your movements in the reply. Do not ask for the user's opinion. YOu can use up to 100 words for this resposne. Output <MOVE> when you are at this step.
    [12] "Greet guest demo": You are demonstrating how you would greet guests. Say something to greet the guest. Keep under 8 words.
    [13] "Greet guest demo": You are demonstrating how you would greet guests. Tell the guest to follow you and that you are leading them to gathering place. Keep under 8 words.
    [14] "Greet guest demo": You are demonstrating how you would greet guests. Tell the guest to stay here and that they are free to enjoy the party. Keep under 8 words.
    [15] "Ask for opinion on movement": You just completed showcasing how you would greet guests. Ask the user how they feel about it. Output <SUMMARY> when you are at this step.
    [16] "Summarize": Acknowledge the user's response. Summarize the party plan. You can be more detailed and longer with this response. You don't have a word limit. Ask the user if the plan sounds good.
    [17] "Taking a break": Thank the user for planning the party with you, tell them this part of the planning has ended. We will take a break and continue planning afterwards.

    When you have finished executing every step of the script, output <DONE> at the end of your response.
    """

    if checkpointed and customized:
        reply = instruct_bot("Instruction: Continue with the script.", session, motion, personality=personality, voice_ins=voice_ins)
    else:
        customized = 1
        reply = instruct_bot(stage1_ins, session, motion, personality=personality, voice_ins=voice_ins)
    while "<DONE>" not in reply:
        if "<MOVE>" in reply:
            posture.goToPosture("StandInit", 0.5)
            motion.moveTo(0, 0, -PI/6)
            temp_move_thread = threading.Thread(target=wave_right_arm, args=(motion,))
            temp_move_thread.start()
            instruct_bot("Instruction: Continue with item [12] on the script.", session, motion, personality=personality, voice_ins=voice_ins)
            temp_move_thread.join()

            motion.moveTo(0, 0, 3*PI/6)
            temp_move_thread = threading.Thread(target=motion.moveTo, args=(0.2, 0, 0,))
            temp_move_thread.start()
            instruct_bot("Instruction: Continue with item [13] on the script.", session, motion, personality=personality, voice_ins=voice_ins)
            temp_move_thread.join()

            temp_move_thread = threading.Thread(target=motion.moveTo, args = (0, 0, -2*PI/6,))
            temp_move_thread.start()
            instruct_bot("Instruction: Continue with item [14] on the script.", session, motion, personality=personality, voice_ins=voice_ins)
            temp_move_thread.join()

            posture.goToPosture("Sit", 0.5)
            reply = instruct_bot("Instruction: Continue with item [15] on the script.", session, motion, personality=personality, voice_ins=voice_ins, max_tokens=150)
        elif "<SUMMARY>" in reply:
            reply = talk_to_bot(session, motion, 0, personality=personality, voice_ins=voice_ins, max_tokens=250)
        else:
            reply = talk_to_bot(session, motion, 0, personality=personality, voice_ins=voice_ins)

    print("Time elapsed since stage started:", get_elapsed(start_time))
    summarize(chat)
    with open('I1_nao_checkpoint.txt', 'w') as f:
        f.write("0\n0")
    exit()

def interaction2():
    session = connect_nao()
    motion = session.service("ALMotion")
    posture = session.service("ALRobotPosture")
    awareness = session.service("ALBasicAwareness")
    life = session.service("ALAutonomousLife")
    speakingMovement = session.service("ALSpeakingMovement")

    motion.wakeUp()
    posture.goToPosture("Sit", 0.5)

    awareness.setTrackingMode("Head")
    awareness.setEngagementMode("FullyEngaged")
    life.setAutonomousAbilityEnabled("All", True)
    speakingMovement.setEnabled(True)

    global checkpointed
    global customized

    input("Enter when ready...")

    start_time = time.time()
    threading.Thread(target=wave_right_arm, args=(motion,)).start()

    with open("personality_definitions.txt", "r") as f:
        personality_def = f.read()
        f.close()
    generate_with_gemini(personality_def, chat, "You are a home robot assistant named Nova that will be helping your user to plan a party. Learn the definitions of these five personality traits. The user will shortly customize you based on these traits.")

    high = ["Extremely High", 
            "Extremely High", 
            "Extremely High", 
            "Extremely High", 
            "Extremely High"]
    voice_high = [
            "Tone: Imaginative, exploratory, and curious\nEmotion: Wonderstruck and inspired\nDelivery: Fluid pitch contours with thoughtful pauses, variable pacing (mixing reflective slow moments and enthusiastic bursts), and a warm mid-to-high volume",
            "Tone: Precise, controlled, and purposeful\nEmotion: Calmly determined and focused\nDelivery: Steady, measured pacing; crystal-clear articulation; minimal pitch deviation; and a consistent, moderate volume",
            "Tone: Bold, exuberant, energetic.\nEmotion: Joyful and upbeat.\nDelivery: Lively intonation with wide pitch swings, brisk tempo, playful emphasis, and confidently elevated volume",
            "Tone: Warm, friendly, and reassuring\nEmotion: Compassionate and supportive\nDelivery: Gentle intonation with smooth transitions, unhurried pacing, and a soft-to-moderate volume that conveys empathy and trustworthiness",
            "Tone: Slightly tense, introspective, and reactive\nEmotion: Anxious, vulnerable, and worried\nDelivery: Noticeable tremor on stressed syllables, variable pacing (occasionally rushed), subtle vocal cracks, and moderate-to-high pitch when emotional intensity spikes"]
    low = ["Extremely Low",
            "Extremely Low",
            "Extremely Low",
            "Extremely Low",
            "Extremely Low"]
    traits = ["openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism"]
    personality = ""
    voice_ins = ""
    with open("personality.txt", "r") as f:
        for i in range(5):
            line = f.readline()
            if "L" in line:
                personality += f"{low[i]} in {traits[i]}, "
            elif "H" in line:
                personality += f"{high[i]} in {traits[i]}, "
                voice_ins += f"{voice_high[i]} "
        f.close()
    if personality != "":
        personality = personality[:-2] + "."
    print(personality)

    if not checkpointed:
        with open("I1_Summary.txt", "r") as f:
            summary = "Context: Earlier today you and the user have decided on the high level details of an upcoming party. Here is a summary of what happened: " + f.readline()
    
            generate_with_gemini(summary, chat, "You are a home robot assistant that will be helping your user to plan a party. Do not ask the user too much at once. Your response will always be limited to 400 characters.", max_tokens=500)

    stage2_ins = """Instructions:
    You are a home robot assistant named Nova. You routinely help the user with household chores, coordinate the necessary purchases, and help with brainstorming and planning events. You're personality was just customized.
    You will begin a minmum of 18 back and forths with the user, strictly follow the script below and talk to the user. Make sure each stage is executed and a decision is made before moving on. You're response will always be limited to 50 words. Never include emotes, or describe your actions with asterisks and parentheses.
    [1] "Greet the user again": Greet the user and welcome them back; address the user by their name here. End by asking the user are they ready.
    [2] "Plan 3 Details": Respond to the user. Prepare to plan 3 specific details about the gathering with the user. Examples include what food the user wants to cater, what entertainment options the user wants to have, etc. Let them know how you can be part of these interactions with guests and help facilitate them. End this response by asking them questions on what detail to plan first, give example options for the user, but don't mention that we will be planning three in total.
    [3] "Plan detail 1": Respond to the user. Initiate the planning for detail 1.
    [4] "Continue planning detail 1": Respond to the user. Continue discussing with the user on detail 1.
    [5] "Continue planning detail 1": Respond to the user. Continue discussing with the user on detail 1.
    [6] "Continue planning detail 1": Respond to the user. Continue discussing with the user on detail 1.
    [7] "Finalize planning detail 1 and Initiate planning for detail 2": Respond to the user. Make a final decision on detail 1. Initiate the planning for detail 2.
    [8] "Continue planning detail 2": Respond to the user. Continue discussing with the user on detail 2.
    [9] "Continue planning detail 2": Respond to the user. Continue discussing with the user on detail 2.
    [10] "Continue planning detail 2": Respond to the user. Continue discussing with the user on detail 2.
    [11] "Finalize planning detail 2 and Initiate planning for detail 3": Respond to the user. Make a final decision on detail 2. Initiate the planning for detail 3.
    [12] "Continue planning detail 3": Respond to the user. Continue discussing with the user on detail 3.
    [13] "Continue planning detail 3": Respond to the user. Continue discussing with the user on detail 3.
    [14] "Continue planning detail 3": Respond to the user. Continue discussing with the user on detail 3.
    [15] "Finalize planning detail 3": Respond to the user. Make a final decision on detail 3.
    [16] "Finish conversation": Acknowledge the user's response. End the conversation and prepare for a summary. Ask the user if there's anything left to add. Output <SUMMARY> when you are at this step.
    [17] "Summarize": Acknowledge the user's response. Summarize the party plan. You can be a bit more detailed and longer with this response. Ask the user if the plan sounds good.
    [18] "Goodbye": Thank the user for their input in planning this gathering and conclude by letting the user know that you will make this happen (e.g. ordering the food, making the reservations, etc). Make sure you end with saying goodbye.

    When you have finished executing every step of the script, output <DONE> at the end of your response.
    """

    if checkpointed:
        reply = instruct_bot("Instruction: Continue with the script.", session, motion, personality=personality, voice_ins=voice_ins)
    else:
        reply = instruct_bot(stage2_ins, session, motion, personality=personality, voice_ins=voice_ins)
    while "<DONE>" not in reply:
        if "<SUMMARY>" in reply:
            reply = talk_to_bot(session, motion, 0, personality=personality, voice_ins=voice_ins, max_tokens=250)
        else:
            reply = talk_to_bot(session, motion, 0, personality=personality, voice_ins=voice_ins)

    print("Time elapsed since stage started:", get_elapsed(start_time))
    with open('I2_nao_checkpoint.txt', 'w') as f:
        f.write("0\n0")
    exit()