import sys
import argparse
import socket
import driver

if __name__ == '__main__':
    pass

# Configure the argument parser
parser = argparse.ArgumentParser(description='Python ML client to connect to the TORCS SCRC server.')

parser.add_argument('--host', action='store', dest='host_ip', default='localhost',
                    help='Host IP address (default: localhost)')
parser.add_argument('--port', action='store', type=int, dest='host_port', default=3001,
                    help='Host port number (default: 3001)')
parser.add_argument('--id', action='store', dest='id', default='SCR',
                    help='Bot ID (default: SCR)')
parser.add_argument('--maxEpisodes', action='store', dest='max_episodes', type=int, default=1,
                    help='Maximum number of learning episodes (default: 1)')
parser.add_argument('--maxSteps', action='store', dest='max_steps', type=int, default=0,
                    help='Maximum number of steps (default: 0)')
parser.add_argument('--track', action='store', dest='track', default=None,
                    help='Name of the track')
parser.add_argument('--stage', action='store', dest='stage', type=int, default=3,
                    help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)')
parser.add_argument('--model', action='store', dest='model_path', default='torcs_driver_model.keras',
                    help='Path to the ML model (default: torcs_driver_model.keras)')
parser.add_argument('--metadata', action='store', dest='metadata_path', default='torcs_driver_metadata.pkl',
                    help='Path to the ML model metadata (default: torcs_driver_metadata.pkl)')
parser.add_argument('--verbose', action='store_true', dest='verbose',
                    help='Enable verbose output')

arguments = parser.parse_args()

# Print summary
print('ML Driver Client for TORCS')
print('Connecting to server host ip:', arguments.host_ip, '@ port:', arguments.host_port)
print('Bot ID:', arguments.id)
print('Maximum episodes:', arguments.max_episodes)
print('Maximum steps:', arguments.max_steps)
print('Track:', arguments.track)
print('Stage:', arguments.stage)
print('Model path:', arguments.model_path)
print('Metadata path:', arguments.metadata_path)
print('*********************************************')

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
except socket.error as msg:
    print('Could not make a socket.', msg)
    sys.exit(-1)

sock.settimeout(1.0)

shutdownClient = False
curEpisode = 0

# Initialize ML driver with model paths
d = driver.MLDriver(arguments.stage, arguments.model_path, arguments.metadata_path)

while not shutdownClient:
    while True:
        print('Sending id to server:', arguments.id)
        buf = (arguments.id + d.init()).encode()
        print('Sending init string to server:', buf.decode())

        try:
            sock.sendto(buf, (arguments.host_ip, arguments.host_port))
        except socket.error as msg:
            print("Failed to send data...Exiting...", msg)
            sys.exit(-1)

        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...", msg)
            continue

        if '***identified***' in buf:
            print('Received:', buf)
            break

    currentStep = 0

    while True:
        buf = None
        try:
            buf, addr = sock.recvfrom(1000)
            buf = buf.decode()
        except socket.error as msg:
            print("Didn't get response from server...", msg)
            continue

        if arguments.verbose and buf is not None:
            print('Received:', buf)

        if buf and '***shutdown***' in buf:
            d.onShutDown()
            shutdownClient = True
            print('Client Shutdown')
            break

        if buf and '***restart***' in buf:
            d.onRestart()
            print('Client Restart')
            break

        currentStep += 1
        if currentStep != arguments.max_steps:
            if buf:
                buf = d.drive(buf)
        else:
            buf = '(meta 1)'

        if arguments.verbose:
            print('Sending:', buf)

        if buf:
            try:
                sock.sendto(buf.encode(), (arguments.host_ip, arguments.host_port))
            except socket.error as msg:
                print("Failed to send data...Exiting...", msg)
                sys.exit(-1)

    curEpisode += 1

    if curEpisode == arguments.max_episodes:
        shutdownClient = True

sock.close()