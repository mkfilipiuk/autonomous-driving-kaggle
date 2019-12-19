kaggle competitions download -c pku-autonomous-driving
unzip -n pku-autonomous-driving.zip -d kaggle_6dvnet_compatible

cp 5.cam kaggle_6dvnet_compatible/camera/  # these are proper intrinsic camera parameters
cd kaggle_6dvnet_compatible
mkdir images
cd images
echo "creating symlinks for images... (takes some time)"
for filepath in ../*_images/*; do filename=${filepath##*/}; ln -s $filepath ${filename%.jpg}_Camera_5.jpg; done;

: '  We are not using masks yet but I guess that is how you would store them
mkdir masks
cd masks
echo "creating symlinks for masks... (takes some time)"
for filepath in ../*_masks/*; do filename=${filepath##*/}; ln -s $filepath ${filename%.jpg}_Camera_5.jpg; done;
'
cd ..
mkdir splits
cd splits 
dir -1 ../train_images | sed 's/.jpg/_Camera_5.jpg/' > train.txt
dir -1 ../test_images  | sed 's/.jpg/_Camera_5.jpg/' >  test.txt

cd ../..