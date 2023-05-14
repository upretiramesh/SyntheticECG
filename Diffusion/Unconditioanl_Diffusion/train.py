from diffusion import GaussianDiffusion, Trainer
from diffusion_model import ECGUnetModel


model = ECGUnetModel(dim = 64, dim_mults = (1, 2, 4, 8)).cuda()


diffusion = GaussianDiffusion(model, channels = 8, timesteps = 500, loss_type = 'l1').cuda()


trainer = Trainer(
    diffusion,
    experiment_name='exp1',
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 200000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True                        # turn on mixed precision
)

# train the model
trainer.train()