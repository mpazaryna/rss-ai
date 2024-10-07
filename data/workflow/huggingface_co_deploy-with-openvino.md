# Optimize and deploy models with Optimum-Intel and OpenVINO GenAI

**Source**: HuggingFace
**Date**: time.struct_time(tm_year=2024, tm_mon=9, tm_mday=20, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=4, tm_yday=264, tm_isdst=0)

No summary available

[Read more](https://huggingface.co/blog/deploy-with-openvino)


## Full Content

Optimize and deploy with Optimum-Intel and OpenVINO GenAI
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
Optimize and deploy models with Optimum-Intel and OpenVINO GenAI
Published
September 20, 2024
Update on GitHub
Upvote
14
+8
AlexKoff88
Alexander
Intel
MrOpenVINO
Yury Gorbachev
guest
katuni4ka
Ekaterina Aidova
guest
sandye51
Ilya Lavrenov
Intel
raymondlo84
Raymond Lo
Intel
helenai
Helena
Intel
echarlaix
Ella Charlaix
Table of Contents
Why Use OpenVINO™ for Edge Deployment
Step 1: Setting Up the Environment
Pre-requisites
Step 2: Exporting Models to OpenVINO IR
Using Python API
Using Command Line Interface (CLI)
Step 3: Model Optimization
Using Python API
Using Command Line Interface (CLI):
Step 4: Deploying with OpenVINO GenAI API
Python API Example
C++ API Example
Customizing Generation Config
Conclusion
Additional Resources
Deploying Transformers models at the edge or client-side requires careful consideration of performance and compatibility. Python, though powerful, is not always ideal for such deployments, especially in environments dominated by C++. This blog will guide you through optimizing and deploying Hugging Face Transformers models using Optimum-Intel and OpenVINO™ GenAI, ensuring efficient AI inference with minimal dependencies.
Table of Contents
Why Use OpenVINO™ for Edge Deployment
Step 1: Setting Up the Environment
Step 2: Exporting Models to OpenVINO IR
Step 3: Model Optimization
Step 4: Deploying with OpenVINO GenAI API
Conclusion
Why Use OpenVINO™ for Edge Deployment
OpenVINO™ was originally developed as a C++ AI inference solution, making it ideal for edge and client deployment where minimizing dependencies is crucial. With the introduction of the GenAI API, integrating large language models (LLMs) into C++ or Python applications has become even more straightforward, with features designed to simplify deployment and enhance performance.
Step 1: Setting Up the Environment
Pre-requisites
To start, ensure your environment is properly configured with both Python and C++. Install the necessary Python packages:
pip install --upgrade --upgrade-strategy eager
"optimum[openvino]"
Here are the specific packages used in this blog post:
transformers==4.44
openvino==24.3
openvino-tokenizers==24.3
optimum-intel==1.20
lm-eval==0.4.3
For GenAI C++ libraries installation follow the instruction
here
.
Step 2: Exporting Models to OpenVINO IR
Hugging Face and Intel's collaboration has led to the
Optimum-Intel
project. It is designed to optimize Transformers models for inference on Intel HW. Optimum-Intel supports OpenVINO as an inference backend and its API has wrappers for various model architectures built on top of OpenVINO inference API. All of these wrappers start from
OV
prefix, for example,
OVModelForCausalLM
. Otherwise, it is similar to the API of 🤗 Transformers library.
To export Transformers models to OpenVINO Intermediate Representation (IR) one can use two options: This can be done using Python’s
.from_pretrained()
method or the Optimum command-line interface (CLI). Below are examples using both methods:
Using Python API
from
optimum.intel
import
OVModelForCausalLM
model_id =
"meta-llama/Meta-Llama-3.1-8B"
model = OVModelForCausalLM.from_pretrained(model_id, export=
True
)
model.save_pretrained(
"./llama-3.1-8b-ov"
)
Using Command Line Interface (CLI)
optimum-cli
export
openvino -m meta-llama/Meta-Llama-3.1-8B ./llama-3.1-8b-ov
The
./llama-3.1-8b-ov
folder will contain
.xml
and
bin
IR model files and required configuration files that come from the source model. 🤗 tokenizer will be also converted to the format of
openvino-tokenizers
library and corresponding configuration files will be created in the same folder.
Step 3: Model Optimization
When running LLMs on the resource constrained edge and client devices, model optimization is highly recommended step. Weight-only quantization is a mainstream approach that significantly reduces latency and model footprint. Optimum-Intel offers weight-only quantization through the Neural Network Compression Framework (NNCF), which has a variety of optimization techniques designed specifically for LLMs: from data-free INT8 and INT4 weight quantization to data-aware methods such as
AWQ
,
GPTQ
, quantization scale estimation, mixed-precision quantization.
By default, weights of the models that are larger than one billion parameters are quantized to INT8 precision which is safe in terms of accuracy. It means that the export steps described above lead to the model with 8-bit weights. However, 4-bit integer weight-only quantization allows achieving a better accuracy-performance trade-off.
For
meta-llama/Meta-Llama-3.1-8B
model we recommend stacking AWQ, quantization scale estimation along with mixed-precision INT4/INT8 quantization of weights using a calibration dataset that reflects a deployment use case. As in the case of export, there are two options on how to apply 4-bit weight-only quantization to LLM model:
Using Python API
Specify
quantization_config
parameter in the
.from_pretrained()
method. In this case
OVWeightQuantizationConfig
object should be created and set to this parameter as follows:
from
optimum.intel
import
OVModelForCausalLM, OVWeightQuantizationConfig
MODEL_ID =
"meta-llama/Meta-Llama-3.1-8B"
quantization_config = OVWeightQuantizationConfig(bits=
4
, awq=
True
, scale_estimation=
True
, group_size=
64
, dataset=
"c4"
)
model = OVModelForCausalLM.from_pretrained(MODEL_ID, export=
True
, quantization_config=quantization_config)
model.save_pretrained(
"./llama-3.1-8b-ov"
)
Using Command Line Interface (CLI):
optimum-cli
export
openvino -m meta-llama/Meta-Llama-3.1-8B --weight-format int4 --awq --scale-estimation --group-size 64 --dataset wikitext2 ./llama-3.1-8b-ov
Note
: The model optimization process can take time as it and applies several methods subsequently and uses model inference over the specified dataset.
Model optimization with API is more flexible as it allows using custom datasets that can be passed as an iterable object, for example, and instance of
Dataset
object of 🤗 library or just a list of strings.
Weight quantization usually introduces some degradation of the accuracy metric. To compare optimized and source models we report Word Perplexity metric measured on the
Wikitext
dataset with
lm-evaluation-harness
project which support both 🤗 Transformers and Optimum-Intel models out-of-the-box.
Model
PPL PyTorch FP32
OpenVINO INT8
OpenVINO INT4
meta-llama/Meta-Llama-3.1-8B
7.3366
7.3463
7.8288
Step 4: Deploying with OpenVINO GenAI API
After conversion and optimization, deploying the model using OpenVINO GenAI is straightforward. The LLMPipeline class in OpenVINO GenAI provides both Python and C++ APIs, supporting various text generation methods with minimal dependencies.
Python API Example
import
argparse
import
openvino_genai
device =
"CPU"
# GPU can be used as well
pipe = openvino_genai.LLMPipeline(args.model_dir, device)
config = openvino_genai.GenerationConfig()
config.max_new_tokens =
100
print
(pipe.generate(args.prompt, config))
To run this example you need minimum dependencies to be installed into the Python enviroment as OpenVINO GenAI is designed to provide a lightweight deployment. You can install OpenVINO GenAI package to the same Python environment or create a separate one to compare the application footprint:
pip install openvino-genai==24.3
C++ API Example
Let's see how to run the same pipilene with OpenVINO GenAI C++ API. The GenAI API is designed to be intuitive and provides a seamless migration from 🤗 Transformers API.
Note
: In the below example, any other available device in your environment can be specified for "device" variable. For example, if you are using an Intel CPU with integrated graphics, "GPU" is be a good option to try with. To check the available devices, you can use ov::Core::get_available_devices method (refer to
query-device-properties
).
#
include
"openvino/genai/llm_pipeline.hpp"
#
include
<iostream>
int
main
(
int
argc,
char
* argv[])
{
std::string model_path =
"./llama-3.1-8b-ov"
;
std::string device =
"CPU"
// GPU can be used as well
ov::genai::LLMPipeline
pipe
(model_path, device);
std::cout << pipe.
generate
(
"What is LLM model?"
, ov::genai::
max_new_tokens
(
256
));
}
Customizing Generation Config
LLMPipeline
also allows specifying custom generation options by means of
ov::genai::GenerationConfig
:
ov::genai::GenerationConfig config;
config.max_new_tokens =
256
;
std::string result = pipe.
generate
(prompt, config);
With the LLMPipieline, users can not only effortlessly leverage various decoding algorithms such as Beam Search but also construct an interactive chat scenario with a Streamer as in the below example. Moreover, one can take advantage of enhanced internal optimizations with LLMPipeline, such as reduced prompt processing time with utilization of KV cache of previous chat history with the chat methods : start_chat() and finish_chat() (refer to
using-genai-in-chat-scenario
).
ov::genai::GenerationConfig config;
config.max_new_tokens =
100
;
config.do_sample =
true
;
config.top_p =
0.9
;
config.top_k =
30
;
auto
streamer = [](std::string subword) {
std::cout << subword << std::flush;
return
false
;
};
// Since the streamer is set, the results will
// be printed each time a new token is generated.
pipe.
generate
(prompt, config, streamer);
And finally let's see how to use LLMPipeline in the chat scenario:
pipe.
start_chat
()
for
(
size_t
i =
0
; i < questions.
size
(); i++) {
std::cout <<
"question:\n"
;
std::
getline
(std::cin, prompt);
std::cout << pipe.
generate
(prompt) << std::endl;
}
pipe.
finish_chat
();
Conclusion
The combination of Optimum-Intel and OpenVINO™ GenAI offers a powerful, flexible solution for deploying Hugging Face models at the edge. By following these steps, you can achieve optimized, high-performance AI inference in environments where Python may not be ideal, ensuring your applications run smoothly across Intel hardware.
Additional Resources
You can find more details in this
tutorial
.
To build the C++ examples above refer to this
document
.
OpenVINO Documentation
Jupyter Notebooks
Optimum Documentation
More Articles from our Blog
Accelerating Protein Language Model ProtST on Intel Gaudi 2
By
juliensimon
July 3, 2024
•
2
Faster assisted generation support for Intel Gaudi
By
haimbarad
June 4, 2024
•
3
Upvote
14
+2
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