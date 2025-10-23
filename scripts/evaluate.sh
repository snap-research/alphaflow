# alphaflow-B-2
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-B-2.pt output_dir=data/generate/alphaflow-B-2-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-B-2.pt output_dir=data/generate/alphaflow-B-2-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true

# meanflow-B-2
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/meanflow-B-2.pt output_dir=data/generate/meanflow-B-2-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/meanflow-B-2.pt output_dir=data/generate/meanflow-B-2-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true

# alphaflow-B-2-cfg
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-B-2-cfg.pt output_dir=data/generate/alphaflow-B-2-cfg-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-B-2-cfg.pt output_dir=data/generate/alphaflow-B-2-cfg-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true

# meanflow-B-2-cfg
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/meanflow-B-2-cfg.pt output_dir=data/generate/meanflow-B-2-cfg-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/meanflow-B-2-cfg.pt output_dir=data/generate/meanflow-B-2-cfg-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true

# alphaflow-XL-2-cfg
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-XL-2-cfg.pt output_dir=data/generate/alphaflow-XL-2-cfg-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-XL-2-cfg.pt output_dir=data/generate/alphaflow-XL-2-cfg-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.custom_t_steps=\[1,0.55,0\] sampling.enable_trajectory_sampling=true sampling.enable_consistency_sampling=true

# alphaflow-XL-2-plus-cfg
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-XL-2-plus-cfg.pt output_dir=data/generate/alphaflow-XL-2-plus-cfg-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/alphaflow-XL-2-plus-cfg.pt output_dir=data/generate/alphaflow-XL-2-plus-cfg-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true sampling.enable_consistency_sampling=true

# meanflow-XL-2-cfg
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/meanflow-XL-2-cfg.pt output_dir=data/generate/meanflow-XL-2-cfg-1-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=1 sampling.enable_trajectory_sampling=true

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/generate.py ckpt.snapshot_path=./snapshots/meanflow-XL-2-cfg.pt output_dir=data/generate/meanflow-XL-2-cfg-2-NFE/ seeds=0-49999 batch_size=32 sampling=recflow sampling.sigma_noise=1 sampling.num_steps=2 sampling.enable_trajectory_sampling=true

# eval
#----------------------------------------------------------------------------

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-B-2-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-B-2-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-B-2-cfg-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-B-2-cfg-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-XL-2-cfg-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-XL-2-cfg-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-XL-2-plus-cfg-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/alphaflow-XL-2-plus-cfg-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/meanflow-B-2-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/meanflow-B-2-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/meanflow-B-2-cfg-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/meanflow-B-2-cfg-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/meanflow-XL-2-cfg-1-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]

torchrun --max-restarts=0 --standalone --nproc-per-node=8 scripts/evaluate.py dataset.resolution=\[1,256,256\] gen_dataset.src=data/generate/meanflow-XL-2-cfg-2-NFE/ gen_dataset.resolution=\[1,256,256\] gen_dataset.label_shape=\[\]
