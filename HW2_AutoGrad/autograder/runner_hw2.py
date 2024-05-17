
# NOTE: If you are on Windows and are having trouble with imports, try to run
# this file from inside the autograder directory.
import  numpy as np
import  sys
from    pathlib import Path
import  argparse

# Allows for the import of mytorch from the dir above (either of the below are identical)
# sys.path.append(str(Path(__file__).parent.parent))
# sys.path.append('.')

sys.path.append(str(Path(__file__).parent.parent))

import  mytorch
from    test_conv import * 
from    test_conv_functional import * 


version = "1.0.1"

tests = [
    {
        'name': '1.1 - Functional Backward - Conv1d',
        'autolab': 'Functional Backward - Conv1d',
        'handler': test_conv1d_backward,
        'value': 5,
    },
    {
        'name': '1.2 - Functional Backward - Conv2d',
        'autolab': 'Functional Backward - Conv2d',
        'handler': test_conv2d_backward,
        'value': 5,
    },
    {
        'name': '2.1 - Conv1d (Autograd) Forward',
        'autolab': 'Conv1d (Autograd) Forward',
        'handler': test_cnn1d_layer_forward,
        'value': 2,
    },
    {
        'name': '2.2 - Conv1d (Autograd) Backward',
        'autolab': 'Conv1d (Autograd) Backward',
        'handler': test_cnn1d_layer_backward,
        'value': 3,
    },
    {
        'name': '2.3 - Conv2d (Autograd) Forward',
        'autolab': 'Conv2d (Autograd) Forward',
        'handler': test_cnn2d_layer_forward,
        'value': 2,
    },
    {
        'name': '2.4 - Conv2d (Autograd) Backward',
        'autolab': 'Conv2d (Autograd) Backward',
        'handler': test_cnn2d_layer_backward,
        'value': 3,
    },
]

if __name__ == '__main__':
    print("Autograder version {}\n".format(version))
    run_tests(tests)
