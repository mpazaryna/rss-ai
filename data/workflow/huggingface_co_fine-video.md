# FineVideo: behind the scenes

**Source**: HuggingFace
**Date**: time.struct_time(tm_year=2024, tm_mon=9, tm_mday=23, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=0, tm_yday=267, tm_isdst=0)

No summary available

[Read more](https://huggingface.co/blog/fine-video)


## Full Content

FineVideo: behind the scenes
Hugging Face
Models
Datasets
Spaces
Posts
Docs
Solutions
Pricing
Log In
Sign Up
Back to Articles
FineVideo: behind the scenes
Published
September 23, 2024
Update on GitHub
Upvote
17
+11
mfarre
Miquel Farré
andito
Andres Marafioti
lewtun
Lewis Tunstall
lvwerra
Leandro von Werra
pcuenq
Pedro Cuenca
thomwolf
Thomas Wolf
Table of Contents
About this blog post
Building the Raw dataset
Filtering YouTube-Commons
Downloading the videos
Keeping dynamic content
Word density filtering
Visual dynamism filtering
Video Categorization
Custom built Taxonomy
Content annotation
Feedback loop taxonomy - content annotation
Contributing descriptive metadata
Long videos & Gemini 1.5 Pro
Content selection
Annotating with Gemini 1.5 Pro and Structured Output with GPT4o
Fine Alignment and anomaly filtering
Future Work
Open video datasets are scarce and therefore slowing down the development of open-source video AI. For this reason we built
FineVideo
, a dataset with 43k videos that span 3.4k hours and are annotated with rich descriptions, narrative details, scene splits, and QA pairs.
FineVideo contains a highly diverse collection of videos and metadata which makes it a good ingredient to train models to understand video content, train diffusion models to generate videos from a text description or train computer vision models using its structured data as input.
Wait, you haven’t seen FineVideo yet? take a look at it through the
dataset explorer page
.
Table of Contents
Table of Contents
About this blog post
Building the Raw dataset
Filtering YouTube-Commons
Downloading the videos
Keeping dynamic content
Word density filtering
Visual dynamism filtering
Video Categorization
Custom built Taxonomy
Content annotation
Feedback loop taxonomy - content annotation
Contributing descriptive metadata
Long videos & Gemini 1.5 Pro
Content selection
Annotating with Gemini 1.5 Pro and Structured Output with GPT4o
Fine Alignment and anomaly filtering
Future Work
About this blog post
In this blog post, we share the technical details and code involved in developing FineVideo: a journey that starts with 1.9M videos in
YouTube-Commons
and ends with 44K videos with all details annotated.
A good way to start is taking a look at the different steps of our journey. Those steps involve content filtering, annotation and output structuring.
FineVideo video filtering and annotation pipeline
In the following sections we discuss each of the steps and provide references to relevant parts of the code. If you prefer to navigate the code directly, take a look at our FineVideo repository on
Github
.
First, let’s have a look how we got an initial list of YouTube videos and how we apply some first filters.
Building the Raw dataset
Our journey starts in
YouTube-Commons
: a collection of audio transcripts of videos shared on YouTube under a CC-By license.  Such project was created and is currently maintained by
PleIAs
as part of their corpus collection projects.
Filtering YouTube-Commons
YouTube Commons contain videos and transcripts in a diverse set of languages, our initial task is about narrowing down the content of YouTube Commons to the same language.
We filter YouTube-Commons for videos in English and at the same time we gather relevant metadata. From this initial filtering, we collect 1.9M videos, their closed captions and metadata.
Below some details on the filters and metadata fields that we keep:
Filters
Field
Filter value
Description
original_language
en
videos in English
transcription_language
en
transcripts in English
Metadata fields
Click to Expand Metadata Fields
Field
Description
acodec
audio codec
age_limit
YouTube age restrictions for the video
categories
YouTube video category
channel
YouTube channel
channel_follower_count
Number of subscribed users to the channel
channel_id
YouTube channel identifier
character_count
Number of characters in the closed caption
comment_count
Number of comments in YouTube
description
YouTube video description
duration_string
Video duration in hh:mm:ss format
license
Video License
like_count
Number of video likes in YouTube
resolution
Pixel resolution of the video in the format Width x Height
tags
YouTube free text tags associated with the video
text
Closed Caption
title
YouTube video title
upload_date
YouTube upload date
vcodec
Video Codec
video_id
YouTube video identifier
view_count
Number of views in YouTube
word_count
Number of words in the closed caption
Code for content filtering and metadata gathering available here [
link
]
Downloading the videos
Once we had a target video list with 1.9M videos, we managed to successfully download 1.8M videos (some of the videos where removed by the channel owners and some changed their permissions).
We explored two different approaches for distributed downloading.
Option 1: Video2dataset
video2dataset is an open-source project [
link
] that focuses on distributed video download, transformation and packaging in different dataset formats. The project natively supports Slurm Workload Manager and therefore we could run it in our CPU cluster.
Source: Video2Dataset GitHub page
As all our cluster instances face internet with the same public IP, we contributed to the project the possibility to specify a proxy to facilitate video downloads. While the feature is not yet merged, you can patch video2dataset with our PR [
link
] to use the proxy capabilities.
Option 2: Cloud batch jobs
Most cloud providers have the possibility to run jobs by simply defining the type of instance that will execute each job, defining a queue and providing a container with the code that will be executed.
We used Google Cloud and AWS to run a custom-made docker container that downloads videos and metadata with
ytdlp
and pushes the results to S3.
The files to build the Docker container can be found here [
code
].
Our conclusion
While Video2Dataset was functional with a proxy and allowed us to do additional processing steps, the requests / second we could do to the proxy became a bottleneck. This made us pivot towards cloud batch jobs.
Keeping dynamic content
In our search for the best videos, we narrowed down our selection to content where there is both visual action and people speaking at a mid-fast pace. We achieve this with word density filtering and visual dynamism filtering.
Word density filtering
We took the density of words in the video as a proxy of audio dynamism. The definition of word density being:
Word density = Number of words in closed captions / Total video length in seconds
By sampling and visually evaluating the quality of the content at different density thresholds, we decided to remove all videos with a word density lower than 0.5 words/second.
Examples:
Word density
Example
0.25
0.5
0.75
1.0
The code to filter by word density and explore examples can be found here [
link
]
Visual dynamism filtering
We repurposed FFMPEG’s
Freezedetect filter
to  judge the dynamism of the video. While this filter is designed to identify frozen sections of a video (multiple equal frames placed one after the other), we could also identify chunks with low movement by exaggerating the
noise
parameter to a very high value.
Rather than running freezedetect across the full video, we analyzed the video by temporal segments and we voted if the video was static based on the amount of segments categorized as static. Through manual evaluation we set a threshold to discard the video if 40% of the segments analyzed have low movement.
Some types of content discarded after this filtering:
Type
Example
Static image with music
Presentation screen cast
Highly static people talking to camera
The DockerFile and code to classify video by its dynamism can be found here [
link
]
From the 1.8M videos analyzed, after this step we keep 600K dynamic videos. At this stage, we dig deeper into the content of the videos, which will be key to ensure diversity in the dataset.
Video Categorization
In order to achieve the most diverse content selection, we categorized the 600K filtered assets using the closed captioning and YouTube metadata. As a way to gain control on the categorization, we created a taxonomy and guided the annotation process to adhere to the taxonomy.
Custom built Taxonomy
We bootstrapped the custom built taxonomy using GPT4-o and an information scientist reviewed and adjusted it. The taxonomy contains 126 fine categories aggregated in multiple levels. This multi-level approach allow users of FineVideo to slice the dataset to fit their particular use-case.
The taxonomy is also available in JSON [
link
]
With an initial version of the taxonomy we started content annotation and by looking at the results of content annotation, with the help of an information scientist, we adjusted the taxonomy accordingly.
Content annotation
We categorized the videos using Llama 3.1 70B served through Text Generation Inference
TGI
[
code
].
The prompt required multiple iterations to ensure the answer is strictly a category in our taxonomy. During our prompt evaluation we learned that by removing the existing YouTube tags and categories from the prompt, the quality of our results increased drastically: YouTube metadata was biasing the text generated by Llama 3.1 towards one of the categories provided by YouTube.
prompt_template =
"""
Given those categories: {leaves}
Classify a youtube video given its closed captioning and some metadata details. RETURN ONLY the selected category and nothing else!
Title: {title}
Description: {description}
Channel: {channel}
Closed Caption: {closed_caption}
"""
Feedback loop taxonomy - content annotation
Taxomy adjustments during content categorization
One of the roles of information scientists is to curate taxonomies over-time to add new categories or add some extra degrees of differentiation when needed.
Using LLMs to categorize content stresses the need to adjust taxonomies from months / years to hours. Furthermore, in some cases, we created categories specifically to discard sensitive videos such as the ones falling under
Firearms & Weapons
and
Substance Use & Drugs
.
Contributing descriptive metadata
At this point of the process, we have three sources of video level metadata:
video category (inferred with Llama 3.1)
YouTube Metadata (title, description)
Transcripts from YouTube-Commons
In order to contribute in the field of video understanding, we decided to go deeper into timecode-level metadata, for example activities, objects, narrative and editing aspects.
While human annotation was something we considered as part of active learning setup where one or more models propose annotations and the human does a QA step, as we will discuss in the next sections, we found in Gemini a good solution especially when we constrained the input video length and the output format.
Long videos & Gemini 1.5 Pro
We dig deeper into Gemini 1.5 Pro iterating our prompt and testing it with different content length.
Given its limitation to 1M tokens, which is approximately equivalent to ~1hour of video, we were forced to drop videos longer than 1 hour.
An idea to overcome this situation was to accelerate videos longer than one hour and that way fit in Gemini’s context.
Exploration: accelerating videos to fit more content in Gemini's context
While it seemed to work at high level, when we started looking at the details we realized that only the first minutes of the video were accurately annotated.
Finding that quality drops on long videos made us wonder: is this an issue impacting the rest of our videos? by sampling videos of different lengths and inspecting the video coverage of the annotations, we found a reduction in quality for videos longer than 10+ minutes.
Aligned with our goal to bring high quality data back to the community, we dropped videos longer than 10+ minutes.
Content selection
Given that each hour of video costs more than $5 to annotate with Gemini, we can’t annotate all the videos that we have after filtering. Therefore, we wanted to make sure that we have a good coverage over all topics and we search a good compromise of content diversity for late-pre-training / fine-tuning task and budget. We set this size constraint to 4K hours of video.
In order to go from 600K videos to 4K hours of content we prepared an algorithm that balances content categories, user engagement, and channel representation to achieve the targeted duration.
Algorithm flow diagram
Some key parts of the content selection algorithm:
Activity Score
: We calculate an engagement metric for each video by combining comment, view, and like counts with weighted importance. This score helps prioritize videos that have resonated well with viewers.
Video Selection
: This step iteratively selects videos to meet the target duration while ensuring diversity. It balances between high-engagement content and representation from various categories and channels, using a penalty system to avoid overrepresentation of any single channel.
Final Adjustment
: We adjust the selection to match the target duration as closely as possible without exceeding it. It sorts the selected videos by duration and adds them to the final list until reaching the closest possible total duration to the target.
The code can be found in the repository
[link]
.
Annotating with Gemini 1.5 Pro and Structured Output with GPT4o
Why structured data?
One of our goals with FineVideo is to provide structured data as a way to empower our community: if you are working on MultiModal LLMs, you can slice the data and decide which categories fit your pre-training or fine-tuning mix. If you are more into computer vision, you can directly use the dataset to train classifiers based on the numerical categories included in FineVideo such as the dynamism score, scene boundaries or audio/video correlation score.
Structured data and Gemini 1.5
Gemini 1.5 Pro allows JSON based outputs by providing a schema. We explored this feature and we quickly realized two issues:
We could not fit our original schema into Gemini because our schema is highly complex
When we tried with slightly simpler schemas -still quite complex- the quality of the Gemini results dropped substantially: most of the scene types of data (characters, activities, props) dropped. We tried splitting the prompt in multiple prompts and matching the different prompts to different parts of the schema without much success.
What we observed completely matched what other researchers experienced: adding concrete schema constraints can decrease performance. (
Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models
).
Our solution relied on generating free text with Gemini 1.5 and add a second processing step to align the results of Gemini with our schema.
The Gemini prompt that we used is the following:
Study the video and provide the following details about the video and the semantic scenes that compose it:
- characterList: a list of characters that appear in the whole video and a visual description that should allow me to identify them just seeing an image of them.
- scenes: a list of the scenes with the following properties:
- start/end timestamps of the scene
- list of all the characters that appear in the scene
- list of all activities and their timestamps
- list of all props and their timestamps
- list of all video editing details and their start/end timestamps. Details include transitions, effects, music as well as suggestions like segments of the scene that could be removed and why
- scene mood with notes on how the visuals, audio and context contribute to it. Use the following taxonomy returning only the name in your answer {"moods":{"Positive":[{"name":"Happy","description":"Feeling joyful, content, or delighted."},{"name":"Excited","description":"Feeling enthusiastic, energetic, or eager."},{"name":"Calm","description":"Feeling peaceful, relaxed, or serene."},{"name":"Grateful","description":"Feeling appreciative or thankful."},{"name":"Proud","description":"Feeling satisfied with one's achievements or the achievements of others."}],"Negative":[{"name":"Sad","description":"Feeling down, unhappy, or sorrowful."},{"name":"Angry","description":"Feeling irritated, frustrated, or furious."},{"name":"Anxious","description":"Feeling nervous, worried, or uneasy."},{"name":"Lonely","description":"Feeling isolated, disconnected, or abandoned."},{"name":"Bored","description":"Feeling uninterested, disengaged, or restless."}],"Neutral":[{"name":"Indifferent","description":"Feeling neither particularly positive nor negative."},{"name":"Content","description":"Feeling satisfied but not overly excited."},{"name":"Curious","description":"Feeling interested or inquisitive without strong emotion."},{"name":"Confused","description":"Feeling uncertain or unclear but without strong negative feelings."},{"name":"Pensive","description":"Feeling thoughtful or reflective without strong emotional engagement."}]}}
- specific  mood changing moments inside the scene, report the timestamp and what we transition from/to in any of the dimensions (visual / auditive)
- scene narrative progression and plot development
- specific narrative moments inside the scene. Report the timestamp and what happened
- character interaction and dynamics descriptions and their start/end timestamps
- specific thematic elements and descriptions
- specific relevant happenings to create deeper meanings and subtexts not explicitly stated that contribute to the richness and depth of the content, timestamp and descriptions
- dynamism score of the scene. Score between 0 and 1. 1 is highly dynamic
- audio - visual correlation score. Score between 0 and 1. 0 what we see is not correlated with the speech and 1 is highly correlated
- storylines: a list of the different storylines found and which scenes belong to it.
- Specify where is the climax (scene and timestamp) and if the content is being presented a narrative story, or is it more like a collection of facts or non-narrative information
- if there are scenes not matching storylines, explain how those scenes contribute to the video
- looking at the overall video and the storylines, which segments of the video could be trimmed to make it more dynamic?
- q&a: a list of 5 questions/answers about the video that focus on fine details (objects and or activities), overall story reasoning and mood. Focus on Q&A aspects captured on the audio and the video whenever possible difficult to get only by looking at the transcription.
Adding Instructor
Once the result was processed by Gemini, we parsed it with
Instructor
: a library built on top of Pydantic to achieve structured outputs given a schema. See table with example below.
Instructor allowed us to experiment with different models to convert the free text from Gemini into the schema that we defined in Pydantic. We tried Gemini and GPT4o and we sticked with GPT4o given the higher success rate.
Video
Gemini Output
Instructor Output
CharacterList:
Man Slim build, brown eyes, shaved sides, black hoodie with colorful logo, black pants.
Scenes
Scene 1
Start 0:00 End 0:55
Characters: [Man]
Activities:
Introduces bus
Describes peaceful location with cows
Props: Bus, cows, deck.
Mood:Excited, adventure.
Narrative Progression:
Introduction to bus.
Tour begins outside, highlighting nature and relaxation.
Dynamism Score 0.7
Audio-Visual Correlation 1
{
"title": "Bertie the Bus Tour",
"description": "Guided tour of converted bus.",
"characterList": [
{
"name": "Narrator",
"description": "Slim build, brown eyes, shaved sides, black hoodie with colorful logo, black pants."
}
],
"scenes": [
{
"sceneId": 1,
"title": "Introduction to Bus",
"timestamps": {
"start": "0:00",
"end": "0:55"
},
"cast": ["Narrator"],
"activities": [
"Narrator speaks in front of bus",
"Shows outdoor deck with chairs, cows nearby."
],
"props": ["Bus", "Deck", "Cows"],
"mood": "Excited, adventure."
}
],
"dynamismScore": 0.7,
"audioVisualCorrelation": 1
}
It is worth highlighting that the content filtering in Gemini dropped some videos as this is something that can happen to you if you use Gemini. In our case, given the amount of content that we were targetting, the total minutes of content that were dropped by Gemini’s filtering was negligible.
The full code to annotate video can be found here [
link
].
Fine Alignment and anomaly filtering
With the videos annotated and the data properly aligned to our schema, we look at the temporal domain of the data and we ensure its alignment with the video: Gemini 1.5 reads video at 1 frame per second and quite frequently videos have 25 - 29 frames per second. In our Fine Alignment we make sure scene boundaries provided by Gemini 1.5 match the correct frames in the video.
We also use this temporal alignment to discard cases were Gemini stopped providing useful data and a part of the video is wrongly annotated. Notice that thanks to dropping all content longer than 10+ minutes earlier in the pipeline, the number of videos with bad quality data was negligible (lower than 0.5%).
Fine metadata - video scene boundary to shot alignment as a mechanism to discard outliers
Link to video alignment code here [
link
]
Future Work
We are currently preparing the training of a multi-modal LLM trained with FineVideo, we plan to share the model weights and training recipe with the community as soon as it is completed.
We are also open to other extensions of FineVideo, speak up and tell us what you would like to see!
More Articles from our Blog
Scaling robotics datasets with video encoding
By
aliberts
August 27, 2024
•
33
Upvote
17
+5
Company
© Hugging Face
TOS
Privacy
About
Jobs
Website
Models
Datasets
Spaces
Pricing
Docs