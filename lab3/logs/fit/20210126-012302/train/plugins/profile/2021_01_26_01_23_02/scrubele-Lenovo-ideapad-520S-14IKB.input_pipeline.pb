	�G��[�?�G��[�?!�G��[�?	Y�:?�)@Y�:?�)@!Y�:?�)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$�G��[�?���#*T�?AoK�3��?Y4��s�?*	�Zd;9p@2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMap�8�Z���?!���Ff�K@)_EF$a�?1y��3ܙD@:Preprocessing2s
<Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map��l#��?!�,�b�9@)g��I}Y�?1V��+@:Preprocessing2�
JIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeats֧�ŝ?!�:`�f&@)r�߅�ٚ?1\��0�3$@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[8]::ConcatenateU�]�o�?!��{M! @)yt#,*�?1�i��/m@:Preprocessing2F
Iterator::ModelU�g$B#�?!�����H(@)��<��?1�-C� @:Preprocessing2S
Iterator::Model::ParallelMap�+e�X�?!zG(�)�@)�+e�X�?1zG(�)�@:Preprocessing2X
!Iterator::Model::ParallelMap::Zip���s���?!�x��lO@)l\��Ϝ�?1U�e	C@:Preprocessing2n
7Iterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch���հ?!�M�~N�@)���հ?1�M�~N�@:Preprocessing2t
=Iterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate^�����?!J�P��@)�@J��~?1�Qj[7@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat�8�ߡ(�?!�J�Q@)A�G��{?1̃o��@:Preprocessing2�
QIterator::Model::ParallelMap::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::RangepA�,_g?!����?)pA�,_g?1����?:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor��C���R?!7ޘ�G�?)��C���R?17ޘ�G�?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[1]::FromTensor�#����N?!v��=�R�?)�#����N?1v��=�R�?:Preprocessing2�
MIterator::Model::ParallelMap::Zip[0]::FlatMap[9]::Concatenate[0]::TensorSlicea2U0*�C?!S��2+��?)a2U0*�C?1S��2+��?:Preprocessing2�
LIterator::Model::ParallelMap::Zip[0]::FlatMap[8]::Concatenate[1]::FromTensor�;P�<�A?!G��`��?)�;P�<�A?1G��`��?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
both�Your program is MODERATELY input-bound because 12.5% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2B24.1 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	���#*T�?���#*T�?!���#*T�?      ��!       "      ��!       *      ��!       2	oK�3��?oK�3��?!oK�3��?:      ��!       B      ��!       J	4��s�?4��s�?!4��s�?R      ��!       Z	4��s�?4��s�?!4��s�?JCPU_ONLY