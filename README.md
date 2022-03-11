## T-REX-WITH-TWIST

This project is a fun modification to the orignal T-REX game that has popped on our screens many a time when the internet connection was down.</n>
In this modification the jump of the t-rex is controlled by **blinking of the eyes** instead of pressing the space bar or arrow keys.</n>

**This project incorporates the ideas of pre-trained HOG detectors for detection of the eye blink and Flask with SocketIO for communication between the client 
javascript game and the server.**

**I have used a second client that uses Javascript to read video from the webcam and send the video frame by frame to the server where
the frame was processed using pre-trained HOG detectors to detect blinking.**

**If blinking was detected, a message was _broadcasted_ and when received by the first client, triggered the _jump_ action.**


##### Click the link below to watch the application in play.
<a href="https://drive.google.com/file/d/1-Anuv3tNl62DBRgsNjl0o2kfL5LXJrhJ/view?usp=sharing">Click for Video</a>


