import recognizer
import model_injector
import listener

if __name__ == '__main__':
    model_injector = model_injector.ModelInjector("model")
    recognizer = recognizer.Recognizer(model_injector.get_models())
    listener = listener.Listener(recognizer)

    output = listener.listen_file("audio/ru_1.wav")
    print(output)

    # while True:
    #     output = listener.listen_microphone()
    #     print(output)
