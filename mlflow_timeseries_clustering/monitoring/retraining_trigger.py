"""
Retraining trigger logic for the MLflow Time Series Clustering Pipeline.

This module implements the RetrainingTrigger class that monitors noise ratios
and determines when automatic retraining should be initiated based on
configurable thresholds and manual commands.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from ..core.config import PipelineConfig
from ..core.exceptions import PipelineError


logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    """Enumeration of possible retraining trigger reasons."""
    NOISE_THRESHOLD_EXCEEDED = "noise_threshold_exceeded"
    MANUAL_COMMAND = "manual_command"
    SCHEDULED_RETRAINING = "scheduled_retraining"
    PERFORMANCE_DEGRADATION = "performance_degradation"


@dataclass
class RetrainingEvent:
    """
    Data class representing a retraining event.

    Attributes:
        timestamp: When the retraining was triggered
        reason: Why the retraining was triggered
        trigger_value: The value that triggered retraining (e.g., noise ratio)
        threshold_value: The threshold that was exceeded (if applicable)
        metadata: Additional metadata about the trigger event
    """
    timestamp: datetime
    reason: TriggerReason
    trigger_value: Optional[float] = None
    threshold_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class RetrainingTrigger:
    """
    Monitors pipeline performance and triggers retraining when necessary.

    This class implements noise ratio threshold monitoring, manual retraining
    commands, and automatic retraining initiation based on configurable criteria.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize the retraining trigger.

        Args:
            config: Pipeline configuration containing thresholds and parameters
        """
        self.config = config
        self.noise_threshold = config.noise_threshold

        # Tracking variables
        self._noise_history: List[Dict[str, Any]] = []
        self._retraining_history: List[RetrainingEvent] = []
        self._last_retraining: Optional[datetime] = None
        self._manual_trigger_requested = False

        # Callback for retraining initiation
        self._retraining_callback: Optional[Callable[[RetrainingEvent], None]] = None

        # Configuration for advanced triggering logic
        self._min_retraining_interval = timedelta(hours=1)  # Minimum time between retrainings
        self._consecutive_threshold_breaches = 3  # Number of consecutive breaches before triggering
        self._current_breach_count = 0

        logger.info("RetrainingTrigger initialized with noise threshold: %.3f",
                   self.noise_threshold)

    def set_retraining_callback(self, callback: Callable[[RetrainingEvent], None]) -> None:
        """
        Set the callback function to be called when retraining is triggered.

        Args:
            callback: Function to call when retraining should be initiated
        """
        self._retraining_callback = callback
        logger.info("Retraining callback registered")

    def monitor_noise_ratio(self, noise_ratio: float, batch_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Monitor noise ratio and trigger retraining if threshold is exceeded.

        Args:
            noise_ratio: Current noise ratio from batch processing
            batch_metadata: Optional metadata about the current batch

        Returns:
            True if retraining was triggered, False otherwise
        """
        timestamp = datetime.now()

        # Record noise ratio in history
        noise_entry = {
            'timestamp': timestamp,
            'noise_ratio': noise_ratio,
            'metadata': batch_metadata or {}
        }
        self._noise_history.append(noise_entry)

        # Keep only recent history (last 100 entries)
        if len(self._noise_history) > 100:
            self._noise_history = self._noise_history[-100:]

        logger.debug("Monitoring noise ratio: %.4f (threshold: %.4f)",
                    noise_ratio, self.noise_threshold)

        # Check if threshold is exceeded
        if noise_ratio > self.noise_threshold:
            self._current_breach_count += 1
            logger.warning(f"Noise threshold exceeded: {noise_ratio:.4f} > {self.noise_threshold:.4f} (breach count: {self._current_breach_count})")

            # Check if we should trigger retraining
            if self._should_trigger_retraining_for_noise(noise_ratio):
                return self._trigger_retraining(
                    reason=TriggerReason.NOISE_THRESHOLD_EXCEEDED,
                    trigger_value=noise_ratio,
                    threshold_value=self.noise_threshold,
                    metadata=batch_metadata
                )
        else:
            # Reset breach count if noise ratio is below threshold
            if self._current_breach_count > 0:
                logger.info("Noise ratio back below threshold, resetting breach count")
                self._current_breach_count = 0

        return False

    def request_manual_retraining(self, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Request manual retraining.

        Args:
            metadata: Optional metadata about the manual retraining request

        Returns:
            True if retraining was triggered, False otherwise
        """
        logger.info("Manual retraining requested")

        return self._trigger_retraining(
            reason=TriggerReason.MANUAL_COMMAND,
            metadata=metadata
        )

    def check_scheduled_retraining(self, schedule_interval: timedelta) -> bool:
        """
        Check if scheduled retraining should be triggered.

        Args:
            schedule_interval: Time interval for scheduled retraining

        Returns:
            True if retraining was triggered, False otherwise
        """
        if self._last_retraining is None:
            return False

        time_since_last = datetime.now() - self._last_retraining

        if time_since_last >= schedule_interval:
            logger.info(f"Scheduled retraining triggered after {time_since_last}")
            return self._trigger_retraining(
                reason=TriggerReason.SCHEDULED_RETRAINING,
                metadata={'schedule_interval': schedule_interval.total_seconds()}
            )

        return False

    def _should_trigger_retraining_for_noise(self, noise_ratio: float) -> bool:
        """
        Determine if retraining should be triggered based on noise ratio.

        Args:
            noise_ratio: Current noise ratio

        Returns:
            True if retraining should be triggered
        """
        # Check minimum interval since last retraining
        if self._last_retraining is not None:
            time_since_last = datetime.now() - self._last_retraining
            if time_since_last < self._min_retraining_interval:
                logger.info(f"Retraining suppressed due to minimum interval: {time_since_last} < {self._min_retraining_interval}")
                return False

        # Check consecutive threshold breaches
        if self._current_breach_count >= self._consecutive_threshold_breaches:
            logger.warning(f"Consecutive threshold breaches reached: {self._current_breach_count} >= {self._consecutive_threshold_breaches}")
            return True

        # Check for severe threshold breach (immediate trigger)
        severe_threshold = self.noise_threshold * 1.5
        if noise_ratio > severe_threshold:
            logger.error("Severe noise threshold breach: %.4f > %.4f, triggering immediate retraining",
                        noise_ratio, severe_threshold)
            return True

        return False

    def _trigger_retraining(self,
                           reason: TriggerReason,
                           trigger_value: Optional[float] = None,
                           threshold_value: Optional[float] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Trigger retraining with the specified reason and parameters.

        Args:
            reason: Reason for triggering retraining
            trigger_value: Value that triggered retraining
            threshold_value: Threshold that was exceeded
            metadata: Additional metadata

        Returns:
            True if retraining was successfully triggered
        """
        timestamp = datetime.now()

        # Create retraining event
        event = RetrainingEvent(
            timestamp=timestamp,
            reason=reason,
            trigger_value=trigger_value,
            threshold_value=threshold_value,
            metadata=metadata or {}
        )

        # Add to history
        self._retraining_history.append(event)

        # Update last retraining timestamp
        self._last_retraining = timestamp

        # Reset breach count
        self._current_breach_count = 0

        logger.info(f"Retraining triggered: reason={reason.value}, trigger_value={trigger_value}, threshold_value={threshold_value}")

        # Call retraining callback if registered
        if self._retraining_callback is not None:
            try:
                self._retraining_callback(event)
                logger.info("Retraining callback executed successfully")
                return True
            except Exception as e:
                logger.error(f"Retraining callback failed: {str(e)}")
                return False
        else:
            logger.warning("No retraining callback registered, retraining not initiated")
            return False

    def get_noise_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get noise ratio history.

        Args:
            limit: Optional limit on number of entries to return

        Returns:
            List of noise ratio history entries
        """
        if limit is None:
            return self._noise_history.copy()
        return self._noise_history[-limit:].copy()

    def get_retraining_history(self, limit: Optional[int] = None) -> List[RetrainingEvent]:
        """
        Get retraining history.

        Args:
            limit: Optional limit on number of entries to return

        Returns:
            List of retraining events
        """
        if limit is None:
            return self._retraining_history.copy()
        return self._retraining_history[-limit:].copy()

    def get_trigger_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about trigger behavior.

        Returns:
            Dictionary containing trigger statistics
        """
        total_retrainings = len(self._retraining_history)

        # Count retrainings by reason
        reason_counts = {}
        for event in self._retraining_history:
            reason = event.reason.value
            reason_counts[reason] = reason_counts.get(reason, 0) + 1

        # Calculate noise ratio statistics
        noise_ratios = [entry['noise_ratio'] for entry in self._noise_history]
        noise_stats = {}
        if noise_ratios:
            import numpy as np
            noise_stats = {
                'mean': float(np.mean(noise_ratios)),
                'std': float(np.std(noise_ratios)),
                'min': float(np.min(noise_ratios)),
                'max': float(np.max(noise_ratios)),
                'current': noise_ratios[-1] if noise_ratios else None
            }

        # Calculate time since last retraining
        time_since_last = None
        if self._last_retraining is not None:
            time_since_last = (datetime.now() - self._last_retraining).total_seconds()

        return {
            'total_retrainings': total_retrainings,
            'retrainings_by_reason': reason_counts,
            'noise_threshold': self.noise_threshold,
            'current_breach_count': self._current_breach_count,
            'consecutive_threshold_breaches': self._consecutive_threshold_breaches,
            'min_retraining_interval_seconds': self._min_retraining_interval.total_seconds(),
            'time_since_last_retraining_seconds': time_since_last,
            'noise_ratio_statistics': noise_stats,
            'total_noise_measurements': len(self._noise_history)
        }

    def update_configuration(self, new_config: PipelineConfig) -> None:
        """
        Update trigger configuration.

        Args:
            new_config: New pipeline configuration
        """
        old_threshold = self.noise_threshold
        self.config = new_config
        self.noise_threshold = new_config.noise_threshold

        if old_threshold != self.noise_threshold:
            logger.info("Noise threshold updated: %.3f -> %.3f",
                       old_threshold, self.noise_threshold)
            # Reset breach count when threshold changes
            self._current_breach_count = 0

    def reset_trigger_state(self) -> None:
        """
        Reset the trigger state (useful for testing or manual reset).
        """
        logger.info("Resetting retraining trigger state")
        self._current_breach_count = 0
        self._manual_trigger_requested = False
        # Note: We don't reset history or last_retraining timestamp
        # as these are important for tracking purposes