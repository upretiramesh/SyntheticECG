
from diffusion_model_condition import ECGUnetCondition
from diffusion_conditional import GaussianDiffusion, Trainer

condition = 'muse'
model = ECGUnetCondition(dim = 64, dim_mults = (1, 2, 4, 8), condition=condition).cuda()
diffusion = GaussianDiffusion(model, channels = 8, timesteps = 500,  loss_type = 'l1').cuda()

trainer = Trainer(
    diffusion,
    experiment_name='exp1',
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 200000,
    gradient_accumulate_every = 2,   
    ema_decay = 0.995,               
    amp = True,
    condition=condition
    )

trainer.train()