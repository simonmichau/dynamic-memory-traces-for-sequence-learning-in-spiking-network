import os

print os.path.dirname(os.path.abspath(__file__))
print os.path.abspath(os.getcwd())
print os.path.dirname(os.path.realpath(__file__))