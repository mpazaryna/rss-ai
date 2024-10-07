# Improving Parquet Dedupe on Hugging Face Hub

**Source**: HuggingFace
**Date**: time.struct_time(tm_year=2024, tm_mon=10, tm_mday=5, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=5, tm_yday=279, tm_isdst=0)

No summary available

[Read more](https://huggingface.co/blog/improve_parquet_dedupe)


## Full Content

Improving Parquet Dedupe on Hugging Face Hub
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
Improving Parquet Dedupe on Hugging Face Hub
Published
October 5, 2024
Update on GitHub
Upvote
5
yuchenglow
yuchenglow
seanses
Di Xiao
Background
Append
Modification
Deletion
Content Defined Row Groups
The Xet team at Hugging Face is working on improving the efficiency of the Hub's
storage architecture to make it easier and quicker for users to
store and update data and models. As Hugging Face hosts nearly 11PB of datasets
with Parquet files alone accounting for over 2.2PB of that storage,
optimizing Parquet storage is of pretty high priority.
Most Parquet files are bulk exports from various data analysis pipelines
or databases, often appearing as full snapshots rather than incremental
updates. Data deduplication becomes critical for efficiency when users want to
update their datasets on a regular basis. Only by deduplicating can we store
all versions as compactly as possible, without requiring everything to be uploaded
again on every update. In an ideal case, we should be able to store every version
of a growing dataset with only a little more space than the size of its largest version.
Our default storage algorithm uses byte-level
Content-Defined Chunking (CDC)
,
which generally dedupes well over insertions and deletions, but the Parquet layout brings some challenges.
Here we run some experiments to see how some simple modifications behave on
Parquet files, using a 2GB Parquet file with 1,092,000 rows from the
FineWeb
dataset and generating visualizations using our
dedupe
estimator
.
Background
Parquet tables work by splitting the table into row groups, each with a fixed
number of rows (for instance 1000 rows). Each column within the row group is
then compressed and stored:
Intuitively, this means that operations which do not mess with the row
grouping, like modifications or appends, should dedupe pretty well. So let's
test this out!
Append
Here we append 10,000 new rows to the file and compare the results with the
original version. Green represents all deduped blocks, red represents all
new blocks, and shades in between show different levels of deduplication.
We can see that indeed we are able to dedupe nearly the entire file,
but only with changes seen at the end of the file. The new file is 99.1%
deduped, requiring only 20MB of additional storage. This matches our
intuition pretty well.
Modification
Given the layout, we would expect that row modifications to be pretty
isolated, but this is apparently not the case. Here we make a small
modification to row 10000, and we see that while most of the file does dedupe,
there are many small regularly spaced sections of new data!
A quick scan of the
Parquet file
format
suggests
that absolute file offsets are part of the Parquet column headers (see the
structures ColumnChunk and ColumnMetaData)! This means that any
modification is likely to rewrite all the Column headers. So while the
data does dedupe well (it is mostly green), we get new bytes in every
column header.
In this case, the new file is only 89% deduped, requiring 230MB of additional
storage.
Deletion
Here we delete a row from the middle of the file (note: insertion should have
similar behavior). As this reorganizes the entire row group layout (each
row group is 1000 rows), we see that while we dedupe the first half of
the file, the remaining file has completely new blocks.
This is mostly because the Parquet format compresses each column
aggressively. If we turn off compression we are able to dedupe more
aggressively:
However the file sizes are nearly 2x larger if we store the data
uncompressed.
Is it possible to have the benefit of dedupe and compression at the same
time?
Content Defined Row Groups
One potential solution is to use not only byte-level CDC, but also apply it at the row level:
we split row groups not based on absolute count (1000 rows), but on a hash of a provided
“Key” column. In other words, I split off a row group whenever the hash of
the key column % [target row count] = 0, with some allowances for a minimum
and a maximum row group size.
I hacked up a quick inefficient experimental demonstration
here
.
With this, we are able to efficiently dedupe across compressed Parquet
files even as I delete a row. Here we clearly see a big red block
representing the rewritten row group, followed by a small change for every
column header.
Optimizing Parquet for Dedupe-ability
Based on these experiments, we could consider improving Parquet file
dedupe-ability in a couple of ways:
Use relative offsets instead of absolute offsets for file structure
data. This would make the Parquet structures position independent and
easy to “memcpy” around, although it is an involving file format change that
is probably difficult to do.
Support content defined chunking on row groups. The format actually
supports this today as it does not require row groups to be uniformly sized,
so this can be done with minimal blast radius. Only Parquet format writers
would have to be updated.
While we will continue exploring ways to improve Parquet storage performance
(Ex: perhaps we could optionally rewrite Parquet files before uploading?
Strip absolute file offsets on upload and restore on download?), we would
love to work with the Apache Arrow project to see if there is interest in
implementing some of these ideas in the Parquet / Arrow code base.
In the meantime, we are also exploring the behavior of our data dedupe process
on other common filetypes. Please do try our
dedupe
estimator
and tell us about
your findings!
Upvote
5
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