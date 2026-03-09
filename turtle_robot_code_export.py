#!/usr/bin/env python3

import os
import sys
import time
import math
import json
import string
import threading
import subprocess
import random
import rospy
import moveit_commander
from geometry_msgs.msg import Twist

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import openai
from openai import OpenAI
from google import genai
from google.genai import types
import point
import multiprocessing

import asyncio
import websockets

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI()

# Loading gemini client
client = genai.Client(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Configuration
TB_IP = "XXXX"
PI_LED_WS = "ws://" + TB_IP + ":XXX"
RECORD_LOCAL = "XXX.wav"
RESPONSE_LOCAL = "XXX.wav"

# global variables for threading
isRecording = False
isSpeaking= False

# Checkpoint
with open("I1_TB_checkpoint.txt", "r") as f:
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


# --- Light functions ---
async def send_led_command(cmd):
    """
    Open a fresh WS connection, send one command, print the Pi's ACK, then close.
    """
    try:
        async with websockets.connect(PI_LED_WS) as ws:
            await ws.send(cmd)
            resp = await ws.recv()
    except Exception as e:
        print("⚠ LED WS error:", e)

async def listening_lights_on():
    await send_led_command("led_listening_start")

async def listening_lights_off():
    await send_led_command("led_listening_stop")

async def thinking_lights_on():
    await send_led_command("led_thinking_start")

async def thinking_lights_off():
    await send_led_command("led_thinking_stop")


# --- Audio functions ---
def record_audio(arm):
    global isRecording
    isRecording = True
    print("Recording audio from mic...")

    nod_thread = threading.Thread(target=random_nodding, args=(arm,))
    nod_thread.start()
    clip_duration = 0.1
    frames_per_clip = int(clip_duration * 16000)
    recorded_chunks = []

    def recording_loop():
        nonlocal recorded_chunks
        try:
            while isRecording:
                clip = sd.rec(frames_per_clip, samplerate=16000, channels=1, dtype='int16')
                sd.wait()
                recorded_chunks.append(clip)
        except Exception as e:
            print("Recording loop error:", e)

    try:
        asyncio.run(listening_lights_on())
        record_thread = threading.Thread(target=recording_loop)
        record_thread.start()

        if input(">> ") == "exit":
            isRecording = False
            asyncio.run(listening_lights_off())
            exit()
        isRecording = False
        record_thread.join()
        asyncio.run(listening_lights_off())
        audio_data = np.concatenate(recorded_chunks)
        os.remove(RECORD_LOCAL)
        write(RECORD_LOCAL, 16000, audio_data)
        print("Recording saved to", RECORD_LOCAL)

    except Exception as e:
        print("Recording error:", e)
        isRecording = False

def play_audio(cmd_pub):
    global isSpeaking
    isSpeaking = True
    asyncio.run(thinking_lights_off())
    motion_thread = threading.Thread(target=random_talking_motions, args=(cmd_pub,))
    motion_thread.start()
    print("Playing audio (fallback)...")
    try:
        subprocess.run(["ffplay", "-nodisp", "-autoexit", RESPONSE_LOCAL], stderr=subprocess.DEVNULL, check=True)
        print("Playback complete.")
    except Exception as e:
        print("Playback error:", e)
    isSpeaking = False

# --- AI functions ---
def transcribe_with_whisper(queue, audio_path): 
    print("Transcribing with Whisper...")
    with open(audio_path, "rb") as f:
        response = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="en"
        )
    queue.put(response.text)
    return

def generate_with_gemini(user_prompt, chat, personality_ins, max_tokens=100):
    full_prompt = user_prompt

    asyncio.run(thinking_lights_on())
    response = chat.send_message(
        full_prompt,
        config=types.GenerateContentConfig(
            # max_output_tokens=max_tokens,
            system_instruction=personality_ins
        )
    )
    try:
        return json.loads(response.text)["msg"]
    except:
        return response.text.strip()
    
def instruct_bot(prompt, cmd_pub, max_tokens=250, personality="", voice_ins=""):
    if personality != "":
        personality_ins = "You are a home robot assistant named Nova that will be helping your user. Your personality is " + personality + " Don't mention this to the user and adjust your speech to reflect this personality.\n"
    else:
        personality_ins = "You are a home robot assistant named Nova that will be helping your user."

    try:
        reply = generate_with_gemini(prompt, chat, personality_ins, max_tokens)
        print("----- Reply -----\n", reply)
    except Exception as e:
        print("Gemini error:", e)
    try:
        ttsSuccess = False
        wait_time = min(max(4, len(reply)//40), 25)
        while not ttsSuccess:
            tts_process = multiprocessing.Process(target=synthesize_with_openai_tts, args=(reply,voice_ins,))
            tts_process.start()
            tts_process.join(timeout=wait_time)
            if tts_process.is_alive():
                tts_process.terminate()
            else:
                ttsSuccess = True
        play_audio(cmd_pub)
    except Exception as e:
        print("TTS/Playback error:", e)
    checkpoint()
    return reply

def talk_to_bot(dont_end_in_question, arm, cmd_pub, max_tokens=250, personality="", voice_ins=""):
    record_audio(arm)
    if personality != "":
        personality_ins = "You are a home robot assistant named Nova that will be helping your user. Your personality is " + personality + " Don't mention this to the user and adjust your speech to reflect this personality.\n"
    else:
        personality_ins = "You are a home robot assistant named Nova that will be helping your user."

    try:
        sttSuccess = False
        results_queue = multiprocessing.Queue()
        while not sttSuccess:
            stt_process = multiprocessing.Process(target=transcribe_with_whisper, args=(results_queue, RECORD_LOCAL,))
            stt_process.start()
            stt_process.join(timeout=5)
            if stt_process.is_alive():
                stt_process.terminate()
            else:
                sttSuccess = True
        user_text = "User: " + results_queue.get()
        print(user_text)
    except Exception as e:
        print("Whisper error:", e)
    try:
        if dont_end_in_question:
            user_text = "Instructions: Do not end this response in a question. Keep it concise.\n" + user_text
        else:
            user_text = "Instructions: Keep the conversation going by ending your response with a question.\n" + user_text
        reply = generate_with_gemini(user_text, chat, personality_ins, max_tokens=max_tokens)
        print("----- Sent -----\n", user_text)
        print("----- Reply -----\n", reply)
    except Exception as e:
        print("Gemini error:", e)
    try:
        a = time.time()
        ttsSuccess = False
        wait_time = min(max(4, len(reply)//40), 25)
        print(wait_time)
        while not ttsSuccess:
            tts_process = multiprocessing.Process(target=synthesize_with_openai_tts, args=(reply,voice_ins,))
            tts_process.start()
            tts_process.join(timeout=wait_time)
            if tts_process.is_alive():
                tts_process.terminate()
            else:
                ttsSuccess = True
        print(time.time() - a)
        play_audio(cmd_pub)
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
    with open('I1_TB_checkpoint.txt', 'w') as f:
        f.write("1\n")
        f.write(str(customized))

def summarize(chat):
    temp = generate_with_gemini("Instructions: Summarize what has happened so far in a neutral and objective tone. You can be detailed and lengthy with your response. This summary will be used to tell another robot what happened during your conversation. Mention only details regarding the user and the party plan, do not include anything regarding the customized personality.", chat, "You will be summarizing this conversation for another robot to read.", max_tokens=250)
    summary = temp.replace("\n", " ")
    with open("I1_Summary.txt", "w") as f:
        f.write(summary)
        f.close()

def synthesize_with_openai_tts(text, ins=""): 
    print("Synthesizing speech...")
    text = text.replace("<DONE>", "")
    text = text.replace("<SUMMARY>", "")
    text = text.replace("<MOVE>", "")

    with openai_client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        instructions=ins
    ) as response:
        with open(RESPONSE_LOCAL, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
            f.close()

# --- Misc Functions ---
def get_elapsed(t):
    return time.time() - t

# --- Movement Functions ---

def move(pub, linear_x=0.0, angular_z=0.0, duration=2.0):
    # Move the base with specified velocities
    twist = Twist()
    twist.linear.x = linear_x
    twist.angular.z = angular_z

    rate = rospy.Rate(10)
    start_time = rospy.Time.now()
    while rospy.Time.now() - start_time < rospy.Duration(duration):
        pub.publish(twist)
        rate.sleep()
    pub.publish(Twist())
    rospy.sleep(1)

def go_safe(arm, joint_goal, name="motion"):
    arm.set_goal_joint_tolerance(0.01)
    arm.set_goal_position_tolerance(0.01)
    arm.set_goal_orientation_tolerance(0.01)

    arm.set_joint_value_target(joint_goal)
    plan = arm.plan()
    if plan and plan[0]:
        success = arm.execute(plan[1], wait=True)
        arm.stop()
        if not success:
            rospy.logwarn(f"Execution failed for {name}")
    else:
        rospy.logwarn(f"Planning failed for {name}")

def perform_wave(arm):

    arm_up = [0, math.radians(-50), 0, 0]
    wave_left = [math.radians(10), math.radians(-50), 0, math.radians(-30)]
    wave_right = [math.radians(-10), math.radians(-50), 0, math.radians(-30)]
    arm_done = [math.radians(0), math.radians(-75), math.radians(60), math.radians(0)]
    go_safe(arm, arm_up, "arm_up_initial")
    rospy.sleep(2.5)

    for i in range(2):
        go_safe(arm, wave_left, f"wave_left {i+1}")
        rospy.sleep(1.9)
        go_safe(arm, wave_right, f"wave_right {i+1}")
        rospy.sleep(1.9)

    go_safe(arm, arm_done, "arm_up_final")
    rospy.sleep(1)

def perform_nod(arm, amount):
    arm_up = [math.radians(0), math.radians(-75), math.radians(60), math.radians(0)]  
    nod_down = [0, math.radians(-75), math.radians(60), math.radians(30)]   
    nod_up = [0, math.radians(-75), math.radians(60), math.radians(0)]   

    for i in range(amount):
        arm.go(nod_up, wait=True)
        rospy.sleep(2)

        arm.go(nod_down, wait=True)
        rospy.sleep(2)

    arm.go(arm_up, wait=True)
    arm.stop()


def random_nodding(arm):
    global isRecording
    while True:
        time.sleep(random.randint(3, 10))
        if not isRecording:
            break
        perform_nod(arm, 1)

def guiding_sequence(arm, cmd_pub, personality="", voice_ins=""):
    move(cmd_pub, angular_z=-0.3925, duration=2.0)

    threading.Thread(target=perform_wave, args=(arm,)).start()
    instruct_bot("Instruction: Continue with item [12] on the script.", cmd_pub, personality=personality, voice_ins=voice_ins)
    rospy.sleep(1)
    threading.Thread(target=point.point_center, args=(arm,)).start()
    instruct_bot("Instruction: Continue with item [13] on the script.", cmd_pub, personality=personality, voice_ins=voice_ins)
    rospy.sleep(1)
    point.point_left(arm)
    point.point_center(arm)
    move(cmd_pub, angular_z=0.3925, duration=6)
    move(cmd_pub, linear_x=0.1, angular_z=0.0, duration=1.5)
    rospy.sleep(1)
    threading.Thread(target=perform_nod, args=(arm, 1,)).start()
    instruct_bot("Instruction: Continue with item [14] on the script.", cmd_pub, personality=personality, voice_ins=voice_ins)
    point.point_center(arm)
    move(cmd_pub, angular_z=-0.3925, duration=4.0)

    rospy.loginfo("Sequence complete.")

def random_talking_motions(cmd_pub):
    global isSpeaking
    rospy.sleep(0.5) 

    fixed_angular_z = 0.2 
    fixed_duration = 1.0 

    time.sleep(random.uniform(2.5, 3.5))

    while isSpeaking:
        time.sleep(random.randint(25, 35) * 0.1)
        if not isSpeaking:
            break
        direction = random.choice(["left", "right"])
        angular_z = fixed_angular_z if direction == "left" else -fixed_angular_z
        move(cmd_pub, angular_z=angular_z, duration=fixed_duration)
        move(cmd_pub, angular_z=-angular_z, duration=fixed_duration)
        time.sleep(random.randint(30, 50) * 0.1)


def interaction1():
    # --------------- Stage 0: Starting Up ---------------
    rospy.init_node("I1_turtlebot", anonymous=True)
    moveit_commander.roscpp_initialize([])
    arm = moveit_commander.MoveGroupCommander("arm")
    arm.set_max_velocity_scaling_factor(1.0)
    arm.set_max_acceleration_scaling_factor(0.7)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

    input("Enter when ready. Remember to start recording...")

    threading.Thread(target=perform_wave, args=(arm,)).start()

    with open("personality.txt", "w") as f:
        f.write("N\nN\nN\nN\nN")
        f.close()

    global checkpointed
    global customized

    start_time = time.time()

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
        reply = instruct_bot("Instruction: Continue with the script.", cmd_pub)
    elif not customized:
        instruct_bot(stage1_ins, cmd_pub)
        ttsSuccess = False
        while not ttsSuccess:
            tts_process = multiprocessing.Process(target=synthesize_with_openai_tts, args=("Just so you know, when you see flashing bars, it means I am listening and recording audio. When you see three dots rotate, it means I am processing information or thinking.",))
            tts_process.start()
            tts_process.join(timeout=8)
            if tts_process.is_alive():
                tts_process.terminate()
            else:
                ttsSuccess = True
        audio_thread = threading.Thread(target=play_audio, args=(cmd_pub,))
        audio_thread.start()
        time.sleep(2)
        asyncio.run(listening_lights_on())
        time.sleep(3.5)
        asyncio.run(listening_lights_off())
        time.sleep(1)
        asyncio.run(thinking_lights_on())
        time.sleep(3)
        asyncio.run(thinking_lights_off())
        audio_thread.join()

        reply = instruct_bot("Instruction: Continue the conversation with item 2 on the script.", cmd_pub)
    if not customized:
        while "<DONE>" not in reply:
            reply = talk_to_bot(0, arm, cmd_pub)

    print("Time elapsed since stage started:", get_elapsed(start_time))

    if input("Waiting for user to finish customizing...") == "exit":
        exit()

    print("Transitioning to Stage 1")

    # --------------- Stage 1: Customizing Personality of Robot ---------------
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
        reply = instruct_bot("Instruction: Continue with the script.", cmd_pub, personality=personality, voice_ins=voice_ins)
    else:
        customized = 1
        reply = instruct_bot(stage1_ins, cmd_pub, personality=personality, voice_ins=voice_ins)
    while "<DONE>" not in reply:
        if "<MOVE>" in reply:
            guiding_sequence(arm, cmd_pub, personality=personality, voice_ins=voice_ins)
            reply = instruct_bot("Instruction: Continue with item [15] on the script.", cmd_pub, personality=personality, voice_ins=voice_ins, max_tokens=250)
        elif "<SUMMARY>" in reply:
            reply = talk_to_bot(0, arm, cmd_pub, personality=personality, voice_ins=voice_ins, max_tokens=250)
        else:
            reply = talk_to_bot(0, arm, cmd_pub, personality=personality, voice_ins=voice_ins)

    print("Time elapsed since stage started:", get_elapsed(start_time))
    summarize(chat)


    moveit_commander.roscpp_shutdown()
    with open('I1_TB_checkpoint.txt', 'w') as f:
        f.write("0\n0")
    asyncio.run(thinking_lights_off())
    asyncio.run(listening_lights_off())
    exit()

def interaction2():
    # --------------- Stage 2: Starting Up ---------------
    rospy.init_node("I2_turtlebot", anonymous=True)
    moveit_commander.roscpp_initialize([])
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    arm = moveit_commander.MoveGroupCommander("arm")
    arm.set_max_velocity_scaling_factor(1.0)
    arm.set_max_acceleration_scaling_factor(0.7)

    input("Enter when ready...")

    wave_thread = threading.Thread(target=perform_wave, args=(arm,))
    wave_thread.start()
    start_time = time.time()

    global checkpointed
    global customized
    
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

    with open("I1_Summary.txt", "r") as f:
        summary = "Context: Earlier today you and the user have decided on the high level details of an upcoming party. Here is a summary of what happened: " + f.readline()
    
    if not checkpointed:
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
        reply = instruct_bot("Instruction: Continue with the script.", cmd_pub, personality=personality, voice_ins=voice_ins)
    else:
        reply = instruct_bot(stage2_ins, cmd_pub, personality=personality, voice_ins=voice_ins)
    while "<DONE>" not in reply:
        if "<SUMMARY>" in reply:
            reply = talk_to_bot(0, arm, cmd_pub, personality=personality, voice_ins=voice_ins, max_tokens=600)
        else:
            reply = talk_to_bot(0, arm, cmd_pub, personality=personality, voice_ins=voice_ins)

    print("Time elapsed since stage started:", get_elapsed(start_time))

    perform_wave(arm)
    moveit_commander.roscpp_shutdown()
    print("Time elapsed since started:", get_elapsed(start_time))
    with open('I2_TB_checkpoint.txt', 'w') as f:
        f.write("0\n0")
    asyncio.run(thinking_lights_off())
    asyncio.run(listening_lights_off())
    exit()
