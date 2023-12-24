cd ./tensorflow/lite/core/api/
echo `pwd` + change flatbuffers/vector.h to flatbuffers/fb_vector.h

sed -i "s@#include \"flatbuffers/vector.h@#include \"flatbuffers/fb_vector.h@g" flatbuffer_conversions.cc

cd ../../../../
cd ./tensorflow/lite/micro/tools/make/downloads/flatbuffers/include/flatbuffers
echo `pwd` + change flatbuffers/*.h to flatbuffers/fb_*.h
for f in *; do
	if [[ "$f" == "flatbuffers.h" || "$f" == "flexbuffers.h" ]]; then
		continue
	fi

	mv "$f" "fb_$f"
	sed -i "s@#include \"flatbuffers/$f@#include \"flatbuffers/fb_$f@g" *
done