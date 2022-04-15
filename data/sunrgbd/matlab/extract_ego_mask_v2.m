% Copyright (c) Facebook, Inc. and its affiliates.
%
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

%% Dump SUNRGBD data to our format
% for each sample, we have RGB image, 2d boxes.
% point cloud (in camera coordinate), calibration and 3d boxes.
%
% Compared to extract_rgbd_data.m in frustum_pointents, use v2 2D and 3D
% bboxes.
%
% Author: Charles R. Qi
%
clear; close all; clc;
addpath(genpath('.'))
addpath('../OFFICIAL_SUNRGBD/SUNRGBDtoolbox/readData')
%% V1 2D&3D BB and Seg masks
% load('./Metadata/SUNRGBDMeta.mat')
% load('./Metadata/SUNRGBD2Dseg.mat')

%% V2 3DBB annotations (overwrites SUNRGBDMeta)
load('../OFFICIAL_SUNRGBD/SUNRGBDMeta3DBB_v2.mat');
load('../OFFICIAL_SUNRGBD/SUNRGBDMeta2DBB_v2.mat');
%% Create folders
mask_folder = '../sunrgbd_trainval/ego_mask/';
mkdir(mask_folder);
%% Read
parfor imageId = 1:10335
    imageId
try
data = SUNRGBDMeta(imageId);
data.depthpath(1:16) = '';
data.depthpath = strcat('../OFFICIAL_SUNRGBD', data.depthpath);
data.rgbpath(1:16) = '';
data.rgbpath = strcat('../OFFICIAL_SUNRGBD', data.rgbpath);

% Write point cloud in depth map
[rgb,points3d,depthInpaint,imsize]=read3dPoints(data);
%rgb(isnan(points3d(:,1)),:) = [];

mask = ~isnan(points3d(:,1))

mat_filename = strcat(num2str(imageId,'%06d'), '.mat');

parsave(strcat(mask_folder, mat_filename), mask);
catch
end

end

function parsave(filename, instance)
save(filename, 'instance');
end
