import urllib
import sys
import urllib2

#urllib.urlretrieve("http://farm4.static.flickr.com/3130/2612813870_28069d38e6.jpg", "Unaltered/"+str(0)+".jpg")

f = open("input.txt")
URLs = f.readlines()
i=733
totalLines = sum(1 for line in open('input.txt')) + 733
for URL in URLs:
    try:
        urllib.urlretrieve(URL, "Unaltered/" + str(i) + ".jpg")
        sys.stdout.write(str(i) + "/" + str(totalLines) + ": (SUCCESS) " + URL)

    except:
        sys.stdout.write(str(i) + "/" + str(totalLines) + ": (FAILURE) " + URL)

    i += 1