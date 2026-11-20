# Private preservation readiness — no transfer performed

On 20 November 2026, read-only anchored-inventory checks again verified:

| Preservation set | Files | Verified bytes | Result |
|---|---:|---:|---|
| Existing private V14/V4/V15 research tree | 98 | 1,353,877,829 | PASS |
| Exact historical public source snapshot | 384 | 6,620,433 | PASS |
| Existing handover/checksum/helper files | 9 indexed files | Not a new payload-size estimate | PASS against existing checksum list |

The private research tree's exact file/directory set, sizes, SHA256s, ownership,
0700 directory modes, 0600 file modes and lack of symlinks passed. The historical
source's exact set, sizes and SHA256s passed. File bytes were hashed opaquely;
no patient records/labels were parsed and no research computation was repeated.
The handover sidecar checks are internal consistency checks; they do not replace
the user's independently retained trusted checksum anchors.

These are current workspace observations, **not evidence of a durable backup**.
The historical source snapshot is in temporary storage. Current GitHub history
does not preserve private artifacts. No encrypted archive, encryption key,
external upload, second durable copy or decrypt/restore drill was created here.
The public table intentionally excludes private paths, inventories, approvals and
patient records. A new aggregate verification receipt is retained privately.

## Concrete preservation set and controls

Use the existing private encrypted-backup/restore plan, inventories and trusted
helpers without rewriting their scientific baseline. Preserve the verified
research tree, exact historical source snapshot and handover together. Also plan
separate coverage for the original inference-v8 five-file bundle and the final
public submission source/locks; those later deliverables are not silently added
to the historical 98/384-file baseline. The original bundle's five identities can
be verified without loading a checkpoint. Keep any supplemental index private.
That supplemental check also passed for all five existing original bundle files
and their owner-only permissions during this review; no weights were loaded.

An encrypted external SSD attached to the user's trusted computer is a simple
user-controlled destination, but this cloud runtime cannot write to an unattached
drive. For a cloud-only workflow, a user-controlled private object-store location
with client-side recipient encryption can be used only after destination and
transfer authorization. Do not use GitHub or a public sharing link for private
payloads. No mounted destination, approved recipient or working transfer route
is established by this review; installed encryption tools alone prove none of these.

Before execution, supply and explicitly approve:

1. Exact destination (private bucket/prefix or verified mounted encrypted volume),
   ownership/access policy and transfer mechanism. Do not provide credentials in chat.
2. An existing user-controlled GPG encryption recipient, authenticated by its
   full fingerprint through a trusted channel. Only an approved public recipient
   certificate is needed for encryption; never send/export a private key or passphrase.
3. The exact preservation scope, including any supplemental original-demo bundle,
   and permission for its encrypted transfer. Do not archive credentials, unrelated
   datasets, historical-final records or the parent workspace.

No key is generated here and no new storage service is selected on the user's behalf.
The historical research/source payload totals 1,360,498,262 bytes before handover,
supplements and archive/encryption overhead. The existing plan reserves at least
4 GiB backup capacity and 8 GiB encrypted restore scratch for that historical scope;
recalculate capacity after any approved scope extension.

## Independent full restoration test

After a later approved backup, record the encrypted archive's actual bytes and
SHA256 plus the trusted inventory/handover anchors in a separate user-controlled
record. A checksum beside its own archive is insufficient as an independent anchor.

1. Select fresh owner-only scratch space on an encrypted filesystem under user
   control; check actual volume identity and available capacity.
2. Verify the ciphertext SHA256 against the separately trusted record **before**
   decryption. Use existing user-controlled decryption access without exposing secrets.
3. Require successful authenticated decryption before decompression/extraction;
   do not stream unauthenticated plaintext directly into tar extraction.
4. Use the hash-verified existing safe extractor. Reject traversal, symlinks,
   hardlinks, devices, duplicates, excessive sizes and overwrites. Restore directories
   0700 and files 0600 into a new destination.
5. Run the anchored inventory verifier on the restored trees. Require all 98 research
   files and 384 historical source files, exact bytes/hashes and entry sets, plus
   any separately approved supplemental bundle/index. Never refresh a baseline to
   hide missing or changed files.
6. Save a private restore receipt with archive/inventory identities, verified counts,
   tool versions, destination reference, time and PASS. No fitting, inference or
   checkpoint loading is part of a restore test.
7. Only then mark the backup verified; retain a separately authorized second encrypted
   copy and keep decryption capability separate. Cloud-session survival is not durability.

Status remains **READY TO PLAN, NOT BACKED UP** until destination, recipient,
transfer and a successful independent restore test are evidenced.
