# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import logging
import numpy as np
import time
import weakref
import torch

import detectron2.utils.comm as comm
from detectron2.structures import Boxes, Instances
from detectron2.layers import cat
from detectron2.utils.events import EventStorage
import copy
import numpy as np

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "SimpleTrainer_CL"]


class HookBase:
    """
    Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:

    .. code-block:: python

        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        hook.after_train()

    Notes:
        1. In the hook method, users can access `self.trainer` to access more
           properties about the context (e.g., current iteration).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly.

    Attributes:
        trainer: A weak reference to the trainer object. Set by the trainer when the hook is
            registered.
    """

    def before_train(self):
        """
        Called before the first iteration.
        """
        pass

    def after_train(self):
        """
        Called after the last iteration.
        """
        pass

    def before_step(self):
        """
        Called before each iteration.
        """
        pass

    def after_step(self):
        """
        Called after each iteration.
        """
        pass


class TrainerBase:
    """
    Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training.
    """

    def __init__(self):
        self._hooks = []

    def register_hooks(self, hooks):
        """
        Register hooks to the trainer. The hooks are executed in the order
        they are registered.

        Args:
            hooks (list[Optional[HookBase]]): list of hooks
        """
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int, cfg = None):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:

            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    print(self.iter)
                    self.before_step()
                    self.run_step(cfg, old_detection = True)
                    self.after_step()
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        for h in self._hooks:
            h.before_step()

    def after_step(self):
        for h in self._hooks:
            h.after_step()
        # this guarantees, that in each hook's after_step, storage.iter == trainer.iter
        self.storage.step()

    def run_step(self):
        raise NotImplementedError


class SimpleTrainer(TrainerBase):
    """
    A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    If you want to do anything fancier than this,
    either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop.
    """

    def __init__(self, model, data_loader, optimizer):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model
        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self, cfg, old_detection = None):
        """
        Implement the standard training logic described above.
        """
        assert cfg is None
        del old_detection
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """
        loss_dict = self.model(data)
        print(loss_dict)
        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        

        # losses.backward()

        # To avoid bad gradients, this is a temporary "hack" and probably very bad thing to do too, come back again to fix!!
        if losses.item() < 50:
            # print("Doing nothing LMAO")
            losses.backward()
            self.optimizer.step()
        else:
            print("THE LOSSS RIGHT NOW ISSSS: ", losses.item())
            # losses.backward()
        
        ## getting rid of the bad way of doing things! Gradient clipping now. 
        # clip_value = 10.0
        # torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value = clip_value)
        # self.optimizer.step()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

class SimpleTrainer_CL(TrainerBase):
    """
    For distilation loss based learning, 
    in forward pass, we first detect previously seen objects.
    We then append it to the groundtruth, 
    then we run the training. 
    This way, we retain old information.

    Author: Dhaivat Bhatt
    """

    def __init__(self, model, data_loader, optimizer, model_old = None):
        """
        Args:
            model: a torch Module. Takes a data from data_loader and returns a
                dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            model_old: Model trained for all previous tasks
        """
        super().__init__()

        """
        We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method.
        """
        model.train()

        self.model = model

        self.model_old = model_old

        ## if old model exists, we subject it to eval mode.
        if self.model_old is not None:
            self.model_old.eval()

        self.data_loader = data_loader
        self._data_loader_iter = iter(data_loader)
        self.optimizer = optimizer

    def run_step(self, cfg, old_detection = False):
        """
        Implement continual learning object detection logic

        Steps:

        1. get the data first
        2. Do the forward pass to detect already seen object classes
        3. 
            i. If it's a deterministic object detector, append them as groundtruth to already seen classes
            ii. If it's a probabilistic object detector, sample from detections and append a few of them.
    
        4. Train the model on currently seen data + old seen classes detected!

        Potential new thing: Add noise to the detected objects.  

        """
        ###

        assert cfg is not None
        
        seen_classes = cfg.CUSTOM_OPTIONS.SEEN_CLASSES
        classifier_type = cfg.CUSTOM_OPTIONS.DETECTOR_TYPE

        ## 1. get the data first ##
        data = next(self._data_loader_iter)

        for i, data_point in enumerate(data):
            data_point['instances'].set('gt_weight', torch.ones(data_point['instances'].gt_classes.size()))
            data[i] = data_point

        if seen_classes and old_detection:
            
            ## 2. Do the forward pass to detect already seen object classes ##
            ## preparing the data for forward pass
            data_dummy = copy.deepcopy(data)
                
            data_final = []
            for data_instance in data_dummy:
                ## doing this so that we get final detections for this shape
                data_dict = {}
                data_dict['image'] = data_instance['image']
                data_dict['height'] = data_dict['image'].shape[1]
                data_dict['width'] = data_dict['image'].shape[2]
                data_final.append(data_dict)

            if self.model_old is None:
                self.model.eval() ## because we are doing detection
                ## predicting detections
                predictions = self.model(data_final)
                ## merging detections from seen classes with groundtruth of new classes
                data = self.merge_gt_and_detections(data, predictions, seen_classes, classifier_type)
                self.model.train()            
            else:
                # print("Getting detections from old model.")
                predictions = self.model_old(data_final)
                ## merging detections from seen classes with groundtruth of new classes
                data = self.merge_gt_and_detections(data, predictions, seen_classes, classifier_type)
        
        ## let's begin boys
        
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        
        data_time = time.perf_counter() - start

        """
        If your want to do something with the losses, you can wrap the model.
        """


        loss_dict = self.model(data)
        print(loss_dict)

        losses = sum(loss for loss in loss_dict.values())
        self._detect_anomaly(losses, loss_dict)

        metrics_dict = loss_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        """
        If you need accumulate gradients or something similar, you can
        wrap the optimizer with your custom `zero_grad()` method.
        """
        self.optimizer.zero_grad()
        
        # To avoid bad gradients, this is a temporary "hack" and probably very bad thing to do too, come back again to fix!!
        if losses.item() < 50:
            # print("Doing nothing LMAO")
            losses.backward()
            self.optimizer.step()
        else:
            print("THE LOSSS RIGHT NOW ISSSS: ", losses.item())
            # losses.backward()
        
        ## getting rid of the bad way of doing things! Gradient clipping now. 
        # clip_value = 10.0
        # torch.nn.utils.clip_grad_value_(self.optimizer.param_groups[0]['params'], clip_value = clip_value)
        # self.optimizer.step()

        """
        If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method.
        """
        
    def merge_gt_and_detections(self, gt, predictions, seen_classes, classifier_type):
        """
        For continual learning setup, it detects all previously seen objects
        and merges them with ground truth.
        TO DO:  
        """

        ## ensure same number of images
        assert len(gt) == len(predictions)

        ## getting groundtruth instances
        gt_instances = [d['instances'] for d in gt]

        ## detected instances
        detected_instances = [p['instances'].to('cpu') for p in predictions]


        ## let's calculate variance and and find inverse variance, normalize between 0.1 to 1.0
        all_inv_variance = []
        min_val = 1e10
        max_val = -1e10
        for i, detectedinstance in enumerate(detected_instances):
            variance_val = []
            for j in range(len(detectedinstance)):
                if detectedinstance[[j]].get('pred_classes').item() in seen_classes and detectedinstance[[j]].get('scores').item() > 0.9:
                    weight = 1. / torch.prod(detectedinstance[[j]].pred_sigma**2).detach().item()
                    variance_val.append(weight)
                    if weight < min_val:
                        min_val = weight
                    if weight > max_val:
                        max_val = weight

            all_inv_variance.append(variance_val)

        #     variance_val = np.array(variance_val)
        #     variance_val = (variance_val - np.min(variance_val)) / np.ptp(variance_val) 


        ## let's go through every image once by one
        for i, detectedinstance in enumerate(detected_instances):

            ## here we append all the ground truth and detected instances for this image
            instance_list = []

            ## let's start with ground truth instance
            instance_list.append(gt_instances[i])
            
            ## append only those detected instances which are from previously seen classes
            for j in range(len(detectedinstance)):

                ## only append if detected instance is from a seen class and it's a confident detection
                if detectedinstance[[j]].get('pred_classes').item() in seen_classes and detectedinstance[[j]].get('scores').item() > 0.90:
                

                    """
                    We attach the detections as they are 
                    if the model being trained is
                    deterministic.

                    """
                    if classifier_type is 'deterministic':
                        ## let's create an instance object for this detection
                        ret = Instances(detectedinstance[[j]].image_size)
                        ret.set('gt_boxes', Boxes(detectedinstance[[j]].get('pred_boxes').tensor.clone().detach())) ## because boxes are expected in the ground truth and not just a tensor
                        ret.set('gt_classes', detectedinstance[[j]].get('pred_classes').clone().detach())
                        instance_list.append(ret)


                    """
                    If the model is probabilistic,
                    then for each detection, we sample a few boxes,
                    assign is the label of the same detection, 
                    and train the model.

                    """
                    if classifier_type is 'probabilistic':
                        ## call function that would sample stuff
                        
                        """
                            use this if we want to sample boxes from the distribution coming
                        """
                        # instance_list = self.sample_boxes(instance_list, detectedinstance[[j]])


                        """
                            use this if you want to weight the sample confidence with
                            inverse covariance
                        """
                        # ret = Instances(detectedinstance[[j]].image_size)
                        # ret.set('gt_boxes', Boxes(detectedinstance[[j]].get('pred_boxes').tensor.clone().detach())) ## because boxes are expected in the ground truth and not just a tensor
                        # ret.set('gt_classes', detectedinstance[[j]].get('pred_classes').clone().detach())
                        # ret.set('gt_weight', torch.ones(1) * (0.5 * (all_inv_variance[i][j] - min_val) / (max_val - min_val) + 0.5 ))
                        # instance_list.append(ret)                        


                        """
                            use this if you want to find smaller and larger
                            bounding box and take the ratio of the two as 
                            weight
                        """

                        gt_weight = self.find_small_large_boxes_ratio(detectedinstance[[j]])
                        ret = Instances(detectedinstance[[j]].image_size)
                        ret.set('gt_boxes', Boxes(detectedinstance[[j]].get('pred_boxes').tensor.clone().detach())) ## because boxes are expected in the ground truth and not just a tensor
                        ret.set('gt_classes', detectedinstance[[j]].get('pred_classes').clone().detach())
                        ret.set('gt_weight', torch.ones(1) * (gt_weight))
                        instance_list.append(ret)                        


            ## for this image size
            image_size = detectedinstance.image_size

            ## let's initialize the instance
            ret = Instances(image_size)


            ## let's go through keys and start merging
            for k in instance_list[0]._fields.keys():

                values = [i.get(k) for i in instance_list]

                v0 = values[0]
                
                if isinstance(v0, torch.Tensor):
                    values = cat(values, dim=0)
                elif isinstance(v0, list):
                    values = list(itertools.chain(*values))
                elif hasattr(type(v0), "cat"):
                    values = type(v0).cat(values)
                else:
                    raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
                ret.set(k, values)
            gt[i]['instances'] = ret          
            
        return gt

    def find_small_large_boxes_ratio(seld, detectedinstance):

        """
            Role of the function:
            It gets the bounding box detection and 
            corresponding uncertainty estimates.
            We calcualte A_small = area(Box - 2sigma) 
            and A_large = area(Box + 2Sigma).
            The ratio is A_ratio = A_small/A_large.
            
            Return:
                gt_weight
        """


        box = detectedinstance.get('pred_boxes').tensor.clone().detach()
        two_sigma = detectedinstance.get('pred_sigma').clone().detach()

        ## calculating areas of the boxes
        box_large = torch.tensor([box[0][0] - two_sigma[0][0], 
            box[0][1] - two_sigma[0][1], box[0][2] + two_sigma[0][2], 
            box[0][3] + two_sigma[0][3]])

        box_small = torch.tensor([box[0][0] + two_sigma[0][0], 
            box[0][1] + two_sigma[0][1], box[0][2] - two_sigma[0][2], 
            box[0][3] - two_sigma[0][3]])
        
        ### let's calculate the area ratio
        area_ratio = Boxes(box_small[None,:]).area()[0] / Boxes(box_large[None,:]).area()[0]

        ### clamp it between 0.5 and 1.0
        area_ratio = torch.clamp(area_ratio, min=0.5, max=1.0).item()

        ### map clamped ratio between 0 to 1. 
        area_ratio = 2 * (area_ratio - 0.5)

        return area_ratio        

    def sample_boxes(self, instance_list, detectedinstance):

        """
            This is used to sample boxes when we have probabilistic object detector
            This would give us more diverse set of boxes. 

        """

        ## TODO: make this a config variable
        num_samples = 5

        for i in range(num_samples):
            sampled_box = torch.distributions.multivariate_normal.MultivariateNormal(detectedinstance.pred_boxes.tensor[0], torch.diag(detectedinstance.pred_sigma[0])).sample()
            sampled_box = Boxes(sampled_box[None])

            ## clip the box to get it within the image
            sampled_box.clip(detectedinstance.image_size)
            ret = Instances(detectedinstance.image_size)
            ret.set('gt_boxes', sampled_box) ## because boxes are expected in the ground truth and not just a tensor
            ret.set('gt_classes', detectedinstance.get('pred_classes').clone().detach())
            ret.set('gt_variance', detectedinstance.get('pred_classes').clone().detach())
            instance_list.append(ret)
        return instance_list

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()
            }
            total_losses_reduced = sum(loss for loss in metrics_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)
