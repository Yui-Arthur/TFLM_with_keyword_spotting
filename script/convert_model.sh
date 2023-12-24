target_file=model_data
while getopts 'p:i:qh' flag; do
case $flag in
    h) 
        echo "-q to set the Quant Model"
        echo "-i to set the Input Model Path"
        echo "-p to set the project dir"
        exit 0
    ;;
    q)
        target_file=model_data_quant
    ;; 
    p)
        project_dir=$OPTARG
    ;;
    m)
        input_model_path=$OPTARG
    ;;
    ?)
        echo "Argument Error"
        exit 1
    ;;
    esac
done

echo $path
xxd -i $input_model_path > $project_dir/model.cc
echo -ne "#include \"$target_file.h\"\nalignas(8)\n" > $project_dir/$target_file.cc
cat $project_dir/model.cc >> $project_dir/$target_file.cc
sed -i -E "s/(unsigned\s.*\s).*(_len|\[\])/const \1${target_file}\2/g" $project_dir/$target_file.cc
rm $project_dir/model.cc