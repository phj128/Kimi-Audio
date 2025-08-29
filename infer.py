from kimia_infer.api.kimia import KimiAudio
import os
import soundfile as sf
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="moonshotai/Kimi-Audio-7B-Instruct")
    args = parser.parse_args()

    model = KimiAudio(
        model_path=args.model_path,
        load_detokenizer=True,
    )

    sampling_params = {
        "audio_temperature": 0.8,
        "audio_top_k": 10,
        "text_temperature": 0.0,
        "text_top_k": 5,
        "audio_repetition_penalty": 1.0,
        "audio_repetition_window_size": 64,
        "text_repetition_penalty": 1.0,
        "text_repetition_window_size": 16,
    }
    output_dir = "test_audios/output"
    os.makedirs(output_dir, exist_ok=True)

    # messages = [
    #     {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/asr_example.wav",
    #     },
    # ]

    # wav, text = model.generate(messages, **sampling_params, output_type="text")
    # print(">>> output text: ", text)

    # audio2audio
    
    text_audio_pairs = [
        ["Hello, can you introduce yourself? And share your with your favorite story?", "my_audios/english_talk.wav"],
        ["Hello, so, do you know anything about basketball?", "my_audios/english_talk2.wav"],
        ["你好, 你好. 请问你能用英语介绍一下你自己吗?", "my_audios/english.wav"],
        ["你好, 请你用英文介绍一下篮球这项运动.", "my_audios/english_2.wav"],
        ["你可以用一个成熟稳重的大叔声音给我讲一个故事吗?", "my_audios/uncle.wav"],
        ["我很难过,你可以用邻家大姐姐的声音安抚我一下吗?", "my_audios/sister.wav"],
        ["你能用非常欢快高兴的男孩的声音给我讲一个趣事吗?", "my_audios/boy.wav"],
        ["你能用一个非常伤心难过的小女孩的声音哭泣地给我讲故事吗?", "my_audios/girl.wav"],
        ["你可以用日语给我打一个招呼并说一下今天的天气吗?", "my_audios/japanese.wav"],
    ]
    
    text_audio_pairs = [
        ["请用悲伤的语气回答我的问题", "my_audios/english_talk.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/english_talk2.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/english.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/english_2.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/uncle.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/sister.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/boy.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/girl.wav"],
        ["请用悲伤的语气回答我的问题", "my_audios/japanese.wav"],
    ]
    
    for text_input, audio_input in text_audio_pairs:
        messages = [
            {"role": "user", "message_type": "text", "content": text_input},
            {
                "role": "user",
                "message_type": "audio",
                "content": audio_input,
            }
        ]

        wav, text = model.generate(messages, **sampling_params, output_type="both")
        sf.write(
            os.path.join(output_dir, f"output_{audio_input.split('/')[-1].split('.')[0]}_ta2ta.wav"),
            wav.detach().cpu().view(-1).numpy(),
            24000,
        )
        print(">>> output text: ", text)
    


    # audio2audio multiturn
    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case1/multiturn_q1.wav",
    #     },
    #     {
    #         "role": "assistant",
    #         "message_type": "audio-text",
    #         "content": ["test_audios/multiturn/case1/multiturn_a1.wav", "当然可以，李白的诗很多，比如这句：“床前明月光，疑是地上霜。举头望明月，低头思故乡。"]
    #     },
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case1/multiturn_q2.wav",
    #     }
    # ]
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # sf.write(
    #     os.path.join(output_dir, "case_1_multiturn_a2.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # print(">>> output text: ", text)


    # messages = [
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case2/multiturn_q1.wav",
    #     },
    #     {
    #         "role": "assistant",
    #         "message_type": "audio-text",
    #         "content": ["test_audios/multiturn/case2/multiturn_a1.wav", "当然可以，这很简单。一二三四五六七八九十。"]
    #     },
    #     {
    #         "role": "user",
    #         "message_type": "audio",
    #         "content": "test_audios/multiturn/case2/multiturn_q2.wav",
    #     }
    # ]
    # wav, text = model.generate(messages, **sampling_params, output_type="both")
    # sf.write(
    #     os.path.join(output_dir, "case_2_multiturn_a2.wav"),
    #     wav.detach().cpu().view(-1).numpy(),
    #     24000,
    # )
    # print(">>> output text: ", text)
