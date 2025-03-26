import os
import shutil
import logging
import gradio as gr
import time

from pydub import AudioSegment  # For checking audio duration

from brain import encode_image, analyze_image_and_query
from voice_patient import record_audio, transcribe_with_groq
from voice_of_chatbot import text_to_speech_with_gtts

# Configure logging for detailed debug output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set your audio file path; adjust as needed.
AUDIO_FILEPATH = r"C:\Users\25ikb\OneDrive\Desktop\Medical_Voicer_chatbot\user.mp3"

# These thresholds can be adjusted:
MIN_FILE_SIZE = 1000      # in bytes (crude file-size check)
MIN_DURATION_SEC = 1.0    # minimal acceptable audio in seconds

def is_valid_audio(file_path):
    """
    Check whether the provided audio file exists, is of sufficient file size,
    and its duration is above a minimal threshold.
    """
    try:
        if not os.path.exists(file_path):
            logging.info("File '%s' does not exist.", file_path)
            return False
        file_size = os.path.getsize(file_path)
        if file_size < MIN_FILE_SIZE:
            logging.warning("File size too small: %d bytes (min %d).", file_size, MIN_FILE_SIZE)
            return False

        # Use pydub to load the audio and check its duration.
        audio = AudioSegment.from_file(file_path)
        duration = audio.duration_seconds
        logging.info("Audio duration: %.2f seconds", duration)
        if duration < MIN_DURATION_SEC:
            logging.warning("Audio duration too short (%.2f sec, min %.2f sec).", duration, MIN_DURATION_SEC)
            return False

        return True
    except Exception as e:
        logging.error("Error validating audio file: %s", e)
        return False

def process_inputs(gradio_audio_filepath, image_filepath):
    """
    Process inputs by:
      1. Using voice input from the Gradio widget if provided and valid;
         otherwise, recording new audio on time using the server microphone.
      2. Transcribing the audio using Groq's STT.
      3. Optionally analyzing an image input.
      4. Converting the chatbot response to speech (TTS) and saving to the same file.
    
    Args:
        gradio_audio_filepath (str): Path to the client's audio file.
        image_filepath (str): Path to the image file (optional).
    
    Returns:
        tuple: (transcription, chatbot response, updated audio file path)
    """
    logging.info("Received inputs - Audio: %s, Image: %s", gradio_audio_filepath, image_filepath)

    # Decide whether to use the client's audio file:
    if gradio_audio_filepath and is_valid_audio(gradio_audio_filepath):
        try:
            logging.info("Using client audio from [%s]. Copying to [%s]...", gradio_audio_filepath, AUDIO_FILEPATH)
            shutil.copy(gradio_audio_filepath, AUDIO_FILEPATH)
            logging.info("Client audio copied successfully.")
        except Exception as e:
            logging.error("Error copying client audio: %s", e)
            logging.info("Falling back to server-side recording...")
            record_audio(file_path=AUDIO_FILEPATH)
    else:
        logging.info("No valid client audio provided. Recording audio on time using server microphone...")
        record_audio(file_path=AUDIO_FILEPATH)

    # Allow a moment for the audio file to be fully written:
    time.sleep(0.5)
    
    if os.path.exists(AUDIO_FILEPATH):
        final_size = os.path.getsize(AUDIO_FILEPATH)
        logging.info("Final audio file '%s' exists; size: %d bytes", AUDIO_FILEPATH, final_size)
    else:
        logging.error("Final audio file does NOT exist at %s", AUDIO_FILEPATH)
        return "Audio file error", "Audio file missing", ""

    # Step 2: Transcribe the audio using Groq STT.
    try:
        speech_to_text_output = transcribe_with_groq(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY"),
            audio_filepath=AUDIO_FILEPATH,
            stt_model="whisper-large-v3"
        )
        logging.info("Transcription complete: %s", speech_to_text_output)
    except Exception as e:
        logging.error("Error transcribing audio: %s", e)
        return "STT error", "", AUDIO_FILEPATH

    # Step 3: Process image input if provided.
    if image_filepath:
        try:
            chatbot_response = analyze_image_and_query(
                query=speech_to_text_output,
                encoded_image=encode_image(image_filepath),
                model="llama-3.2-11b-vision-preview"
            )
            logging.info("Image processed; chatbot response: %s", chatbot_response)
        except Exception as e:
            logging.error("Error processing image input: %s", e)
            chatbot_response = "Error analyzing image."
    else:
        chatbot_response = "No image provided for analysis"
        logging.info("No image provided; using default message.")

    # Step 4: Convert chatbot response to speech (TTS).
    try:
        text_to_speech_with_gtts(input_text=chatbot_response, output_filepath=AUDIO_FILEPATH)
        logging.info("TTS conversion complete. Updated audio saved to %s", AUDIO_FILEPATH)
    except Exception as e:
        logging.error("Error during TTS conversion: %s", e)
        return speech_to_text_output, "TTS error", AUDIO_FILEPATH

    if os.path.exists(AUDIO_FILEPATH):
        final_size = os.path.getsize(AUDIO_FILEPATH)
        logging.info("After TTS, final audio file '%s' exists; size: %d bytes", AUDIO_FILEPATH, final_size)
    else:
        logging.error("Final audio file missing after TTS conversion.")

    return speech_to_text_output, chatbot_response, AUDIO_FILEPATH

# Create the Gradio interface with both audio and image inputs.
iface = gr.Interface(
    fn=process_inputs,
    inputs=[
        gr.Audio(type="filepath", label="Voice Input"),
        gr.Image(type="filepath", label="Image Input (Optional)")
    ],
    outputs=[
        gr.Textbox(label="Speech to Text"),
        gr.Textbox(label="Chatbot Response"),
        gr.Audio(label="Voice Output")
    ],
    title="ChatBot Vision and Voice"
)

iface.launch(debug=True)
