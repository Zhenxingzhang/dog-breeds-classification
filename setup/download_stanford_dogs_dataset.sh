#!/usr/bin/env bash

mkdir -p /data/dog_breeds/stanford_ds
curl -o /data/dog_breeds/stanford_ds/images.tar http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
tar xvf /data/dog_breeds/stanford_ds/images.tar -C /data/dog_breeds/stanford_ds/
curl -o /data/dog_breeds/stanford_ds/annotation.tar http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar
tar xvf /data/dog_breeds/stanford_ds/annotation.tar -C /data/dog_breeds/stanford_ds/
