# Setup Python3.10 with Bluetooth support
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt-get update
sudo apt-get install python3.10 python3.10-venv python3.10-dev

# Install system-level bluetooth dependencies for nxbt
sudo apt-get install libdbus-1-dev libdbus-glib-1-dev

# Setup Project
cd ~/tools
mkdir splat-pp
cd splat-pp
mkdir img macro

# Create and Activate Virtual Environment
python3.10 -m venv venv
source venv/bin/activate 

# Install nxbt 
pip install nxbt 

# Test it
On the Switch, go to the Change Grip Order menu, then run
sudo /venv/bin/nxbt demo
  Running Demo...
  Finished!
It should run through the settings menu then complete with a "Finished!" message.

# Macro Maker
pip install pillow numpy

##Usage: 
`python img2splat.py [input_image] [output_macro.txt] [--mode=snake|smart|typewriter] [--test-pattern=circle|square|lines] [--wait-time=0.05] [--debug]`

```
# Use smart mode (default) for fastest drawing <br>
python img2splat.py my_image.png

# Generate a test pattern <br>
python img2splat.py --test-pattern circle output_macro.txt

# Use snake pattern with custom timing <br>
python img2splat.py my_image.png output.txt --mode snake --wait-time 0.03
```


# Create post
Go to the mailbox in Splatoon 3
Set the pencil to the smallest and position the cursor in the top left corner.
Press the pairing button on top of the controller to disconnect it
When the controller pairing menu shows up, run the macro:
sudo venv/bin/nxbt macro -c img/image_macro.txt -r

It should connect as a pro controller, close the pairing menu, then start drawing. 
It is slow.



