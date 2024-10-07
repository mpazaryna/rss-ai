# Introducing the SQL Console on Datasets

**Source**: HuggingFace
**Date**: time.struct_time(tm_year=2024, tm_mon=9, tm_mday=17, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=261, tm_isdst=0)

No summary available

[Read more](https://huggingface.co/blog/sql-console)


## Full Content

Introducing the SQL Console on Datasets
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
Introducing the SQL Console on Datasets
Published
September 17, 2024
Update on GitHub
Upvote
17
+11
cfahlgren1
Caleb Fahlgren
Introducing the SQL Console for Datasets
How it works
Parquet Conversion
DuckDB WASM 🦆
Example: Converting a dataset from Alpaca to conversations
SQL
SQL Snippets
Resources
Datasets use has been exploding and Hugging Face has become the default home for many datasets. Each month, as the amount of datasets uploaded to the Hub increases, so does the need to query, filter and discover them.
Datasets created on Hugging Face Hub each month
We are very excited to announce that you can now run SQL queries on your datasets directly in the Hugging Face Hub!
Introducing the SQL Console for Datasets
On every public dataset you should see a new
SQL Console
badge. With just one click you can open a SQL Console to query that dataset.
Querying the Magpie-Ultra dataset for excellent, high quality reasoning instructions.
All the work is done in the browser and the console comes with a few neat features:
100% Local
: The SQL Console is powered by
DuckDB
WASM, so you can query your dataset without any dependencies.
Full DuckDB Syntax
: DuckDB has
full SQL
syntax support, along with many built in functions for regex, lists, JSON, embeddings and more. You'll find DuckDB syntax to be very similar to PostgreSQL.
Export Results
: You can export the results of your query to parquet.
Shareable
: You can share your query results of public datasets with a link.
How it works
Parquet Conversion
Most datasets on Hugging Face are stored in Parquet, a columnar data format that is optimized for performance and storage efficiency. The Dataset Viewer on Hugging Face and the SQL Console load the data directly from the datasets Parquet files. And if the dataset is in another format, the first 5GB is auto-converted to Parquet. You can find more information about the Parquet conversion process in the
Dataset Viewer Parquet API documentation
.
Using the Parquet files, the SQL Console creates views for you to query based on your dataset splits and configs.
DuckDB WASM 🦆
DuckDB WASM
is the engine that powers the SQL Console. It is an in-process database engine that runs on Web Assembly in the browser. No server or backend needed.
By running solely in the browser, it gives the user the upmost flexibility to query data as they please without any dependencies. It also makes it really simple to share reproducible results with a simple link.
You may be wondering,
"Will it work for big datasets?"
and the answer is, "Yes!".
Here's a query of the
OpenCo7/UpVoteWeb
dataset which has
12.6M
rows in the Parquet conversion.
You can see we received results for a simple filter query in under 3 seconds.
While queries will take longer based on the size of the dataset and query complexity, you will be surprised about how much you can do with the SQL Console.
As with any technology, there are limitations.
The SQL Console will work for a lot of queries. However, the memory limit is ~3GB, so it is possible to run out of memory and not be able to process the query (
Tip: try to use filters to reduce the amount of data you are querying along with
LIMIT
).
While DuckDB WASM is very powerful, it doesn't have full feature parity with DuckDB. For example, DuckDB WASM does not yet support the
hf://
protocol to query datasets
.
Example: Converting a dataset from Alpaca to conversations
Now that we've introduced the SQL Console, let's explore a practical example. When fine-tuning a Large Language Model (LLM), you often need to work with different data formats. One particularly popular format is the conversational format, where each row represents a multi-turn dialogue between a user and the model. The SQL Console can help us transform data into this format efficiently. Let's see how we can convert an Alpaca dataset to a conversational format using SQL.
Typically, developers would tackle this task with a Python pre-processing step, but we can show how to use the SQL Console to achieve the same in less than 30 seconds.
In the dataset above, click on the
SQL Console
badge to open the SQL Console. You should see the query below automatically populated.
When you are ready, click the
Run Query
button to execute the query.
SQL
-- Convert Alpaca format to Conversation format
WITH
source_view
AS
(
SELECT
*
FROM
train
-- Change 'train' to your desired view name here
)
SELECT
[
struct_pack(
"from" :
=
'user'
,
"value" :
=
CASE
WHEN
input
IS
NOT
NULL
AND
input
!=
''
THEN
instruction
||
'\n\n'
||
input
ELSE
instruction
END
),
struct_pack(
"from" :
=
'assistant'
,
"value" :
=
output
)
]
AS
conversation
FROM
source_view
WHERE
instruction
IS
NOT
NULL
AND
output
IS
NOT
NULL
;
In the query we use the
struct_pack
function to create a new STRUCT row for each conversation.
DuckDB has great documentation on the
STRUCT
Data Type
and
Functions
. You'll find many datasets contain columns with JSON data. DuckDB provides functions to easily parse and query these columns.
Once we have the results, we can download them as a Parquet file. You can see what the final output looks like below.
Try it out!
As an another example, you can try a SQL Console query for
SkunkworksAI/reasoning-0.01
to see instructions with more than 10 reasoning steps.
SQL Snippets
DuckDB has a ton of use cases that we are still exploring. We created a
SQL Snippets
space to showcase what you can do with the SQL Console.
Here are some really interesting use cases we have found:
Filtering a function calling dataset for a specific function with regex
Finding the most popular base models from open-llm-leaderboard
Converting an alpaca dataset to a conversational format
Performing similarity search with embeddings
Filtering 50k+ rows from a dataset for the highest quality, reasoning instructions
Remember, it's one click to download your SQL results as a Parquet file and use for your dataset!
We would love to hear what you think of the SQL Console and if you have any feedback, please comment in this
post!
Resources
DuckDB WASM
DuckDB Syntax
DuckDB WASM Paper
Intro to Parquet Format
Hugging Face + DuckDB
SQL Snippets Space
More Articles from our Blog
FineVideo: behind the scenes
By
mfarre
September 23, 2024
•
17
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