# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
from dataclasses import dataclass, field

import torch
from fairseq import utils
from fairseq.logging import metrics
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion, label_smoothed_nll_loss

@dataclass
class PerExampleLossCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")




@register_criterion(
    "per_example_loss", dataclass=PerExampleLossCriterionConfig
)
class PerExampleLossCriterion(FairseqCriterion):
    """
    Basically Smoothed CE but logs per example loss and accuracy to CSV
    """
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample."""
        net_output = model(**sample["net_input"])

        with torch.no_grad():
            loss, per_token_nll_loss = self.compute_loss(model, net_output, sample, reduce=False)

        # Calculate unreduced NLL loss per example by summing per-token losses
        per_example_nll_loss = per_token_nll_loss.sum(dim=-1)
        
        # Check if we should ignore the prefix
        if self.ignore_prefix_size > 0:
            per_example_nll_loss = per_example_nll_loss[self.ignore_prefix_size:]
        
        # Convert to a list to be stored in logging_output
        per_example_nll_loss_list = per_example_nll_loss.tolist()

        # Compute per-example accuracy
        per_example_accuracy_list = self.compute_per_example_accuracy(model, net_output, sample)
        
        # The loss for backpropagation should be the standard total loss
        total_loss, _ = self.compute_loss(model, net_output, sample, reduce=True)
        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": total_loss.data,
            "nll_loss": total_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "per_example_nll_loss": per_example_nll_loss_list,
            "per_example_accuracy": per_example_accuracy_list,
            "example_ids": sample['id'].tolist(),
        }

        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)

        return total_loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
            target = target[:, self.ignore_prefix_size :].contiguous()
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total


    def compute_per_example_accuracy(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        
        # Calculate accuracy for each sentence in the batch
        per_example_accuracy = []
        for i in range(target.size(0)):
            # Get the target and prediction for the current sentence
            sentence_target = target[i]
            sentence_lprobs = lprobs[i]

            if self.ignore_prefix_size > 0:
                sentence_target = sentence_target[self.ignore_prefix_size:]
                sentence_lprobs = sentence_lprobs[self.ignore_prefix_size:]

            # Create a mask for non-padding tokens
            mask = sentence_target.ne(self.padding_idx)
            
            # Get the total number of non-padding tokens
            total_tokens = torch.sum(mask).item()
            
            if total_tokens == 0:
                per_example_accuracy.append(0.0)
                continue

            # Compare predictions to target
            n_correct = torch.sum(sentence_lprobs.argmax(dim=-1).masked_select(mask).eq(sentence_target.masked_select(mask))).item()
            
            per_example_accuracy.append(n_correct / total_tokens)

        return per_example_accuracy

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training and save to file."""
        # Call the base class's reduce_metrics to get standard log outputs
        LabelSmoothedCrossEntropyCriterion.reduce_metrics(logging_outputs)

        # Custom aggregation for per-example NLL losses
        all_losses = [log['per_example_nll_loss'] for log in logging_outputs if 'per_example_nll_loss' in log]
        all_example_ids = [log['example_ids'] for log in logging_outputs if 'example_ids' in log]

        flattened_losses = [item for sublist in all_losses for item in sublist]
        flattened_ids = [item for sublist in all_example_ids for item in sublist]

        # Custom aggregation for per-example accuracy
        all_accuracies = [log['per_example_accuracy'] for log in logging_outputs if 'per_example_accuracy' in log]
        flattened_accuracies = [item for sublist in all_accuracies for item in sublist]

        log_file_path = f"early_dynamics_output.csv"
        
        is_first_write = not os.path.exists(log_file_path)
        with open(log_file_path, 'a') as f:
            if is_first_write:
                f.write('example_id,nll_loss,accuracy\n')

            for example_id, nll_loss, accuracy in zip(flattened_ids, flattened_losses, flattened_accuracies):
                f.write(f'{example_id},{nll_loss},{accuracy}\n')


    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`.
        """
        # We need to retain the per-example information, so we can't sum the outputs.
        return False