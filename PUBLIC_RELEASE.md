# Public source and model release

This independent Mahjong application is released under AGPL-3.0-only. The source includes the mobile frontend, upload gateway, inference adapters, training, evaluation, data preparation, tests and deployment tools. The selected model weights and their exact hashes are available in the [model release](https://github.com/XTTTZ/mahjong-recognizer/releases/tag/model-v4-20261027). See LICENSE and NOTICE for attribution.

This is a clean snapshot with no private repository history, credentials, raw photographs, SSH keys, experiment screenshots or private run logs. Machine/account placeholders must be configured on a new installation. Poker and Avalon are separate projects and are not included. Historical machine-specific receipts are omitted; regenerate plans after configuring your installation. Data comes from the attributed public source; see docs/data for the recorded preparation and split manifests.

The online service uses the same inference source and exact model weights. The model is experimental: the locked validation winner scored 72/82 automatic exact hands (at least 10 tiles); its one-time sealed test scored 31/47. These source-group holdouts do not establish accuracy in unfamiliar real-world rooms. Users must review every tile. The classifier was recovered from an intact checkpoint after a time budget expired; this is not a claim of training convergence.

A future 12-hour experiment queue is prepared separately and is not automatically launched by this release. Published commands require explicit local submission. No paid license was purchased.
