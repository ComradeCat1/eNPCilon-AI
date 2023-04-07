import tensorflow as tf
import time

one_step_reloaded = tf.saved_model.load('one_step')

def write_response(max_length: int):
    # Check its architecture
    states = None
    next_char = tf.constant(['USER: Hi eNPCilon, how are you?\n\nASSISTANT: '])
    result = [next_char]
    for n in range(max_length):
        next_char, states = one_step_reloaded.generate_one_step(next_char, states=states)
        result.append(next_char)
        result_string = tf.strings.join(result)[0].numpy().decode("utf-8")
        print(result_string[-1], end="")
        time.sleep(0.1)
        if result_string.endswith("USER:"):
            break

write_response(1024)