
import os
import sys
import subprocess
from subprocess import CalledProcessError
import shlex
from retrying import retry
from sagemaker_inference import model_server
import logging

logger = logging.getLogger("entrypoint")


def _retry_if_error(exception):
    return isinstance(exception, CalledProcessError or OSError)


@retry(stop_max_delay=1000 * 50,
    retry_on_exception=_retry_if_error)
def _start_mms():
    # by default the number of workers per model is 1, but we can configure it through the
    # environment variable below if desired.
    # os.environ['SAGEMAKER_MODEL_SERVER_WORKERS'] = '2'

    handler = os.environ.get('HANDLER_SERVICE')
    logger.info(f'Handler: {handler}')
    model_server.start_model_server(handler_service=handler)

def main():
    if sys.argv[1] == 'serve':
        _start_mms()
    else:
        subprocess.check_call(shlex.split(' '.join(sys.argv[1:])))

    # prevent docker exit
    subprocess.call(['tail', '-f', '/dev/null'])


if __name__ == "__main__":
    # model_server.start_model_server(handler_service=os.environ.get('HANDLER_SERVICE'))
    main()
