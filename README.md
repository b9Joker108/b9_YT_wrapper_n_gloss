# Personal and programmable YouTube algorithm, dashboard and knowledge graph visualisation and tooling #

I have had it with the unintelligent, unprogrammable, timewasting blackbox that is the YouTube recommendation algorithm. It does not furnish my requirements. This project is to create a personal workaround. The project is a fork of the formative work in Python done by Chris Lovejoy, refer [post on Chris' weblog](https://chrislovejoy.me/youtube-algorithm). My project goals are different to Chris' as may be garnered from my [project_weblog](/project_weblog/project_weblog.md). Given the explosion of GenAI generated content on YouTube, the lion's share of which I consider signal noise, I need a way to actively filter out this guff and excavate and foreground what I value.



# Following is from README.md of Chris' project

## Setup

### YouTube-API-Key
You will need to acquire a YouTube v3 API key, which you can do so easily [here](https://console.developers.google.com/cloud-resource-manager). A helpful video outlining the process can be found [here](https://www.youtube.com/watch?v=-QMg39gK624). After obtaining the API key, enter it as a string into the [config.yaml file](https://github.com/chris-lovejoy/YouTube-video-finder/blob/master/config.yaml).

### Packages
All requirements are contained within [requirements.txt](https://github.com/chris-lovejoy/YouTube-video-finder/blob/master/requirements.txt).

To install them, execute the following from the root directory:
```
pip install -r requirements.txt
```

## Execution
After configuring config.yaml and installing requirements, the function can be executed from the command line using:

```
python3 main.py 'search term 1' 'search term 2'
```

The default search period is 7 days, but this can be modified with the '--search-period' argument.

For example:

```
python3 main.py 'machine learning' 'medical school' --search-period 10
```

This will call the [**main.py function**](https://github.com/chris-lovejoy/YouTube-video-finder/blob/master/main.py) and output will be printed into the console.
