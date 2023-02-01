import jax
import jax.numpy as jnp

import haiku as hk

__all__ = ["AllConv"]

class AllConv(hk.Module):
    def __init__(self, n_classes, name='AllConv'):
        super().__init__(name=name)
        self.conv1 = hk.Conv2D(
            96, 3, 1, padding='SAME', name='conv1', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/27**0.5, 1/27**0.5)
        )
        self.bn1 = hk.BatchNorm(True, True, 0.9, name='batchnorm1')
        self.conv2 = hk.Conv2D(
            96, 3, 1, padding='SAME', name='conv2', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/864**0.5, 1/864**0.5)
        )
        self.bn2 = hk.BatchNorm(True, True, 0.9, name='batchnorm2')
        self.conv3 = hk.Conv2D(
            96, 3, 2, padding='SAME', name='conv3', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/864**0.5, 1/864**0.5)
        )
        self.bn3 = hk.BatchNorm(True, True, 0.9, name='batchnorm3')
        self.conv4 = hk.Conv2D(
            192, 3, 1, padding='SAME', name='conv4', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/864**0.5, 1/864**0.5)
        )
        self.bn4 = hk.BatchNorm(True, True, 0.9, name='batchnorm4')
        self.conv5 = hk.Conv2D(
            192, 3, 1, padding='SAME', name='conv5', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/1728**0.5, 1/1728**0.5)
        )
        self.bn5 = hk.BatchNorm(True, True, 0.9, name='batchnorm5')
        self.conv6 = hk.Conv2D(
            192, 3, 2, padding='SAME', name='conv6', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/1728**0.5, 1/1728**0.5)
        )
        self.bn6 = hk.BatchNorm(True, True, 0.9, name='batchnorm6')
        self.conv7 = hk.Conv2D(
            192, 3, 1, padding='VALID', name='conv7', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/1728**0.5, 1/1728**0.5)
        )
        self.bn7 = hk.BatchNorm(True, True, 0.9, name='batchnorm7')
        self.conv8 = hk.Conv2D(
            192, 1, 1, padding='VALID', name='conv8', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/192**0.5, 1/192**0.5)
        )
        self.bn8 = hk.BatchNorm(True, True, 0.9, name='batchnorm8')
        self.conv9 = hk.Conv2D(
            n_classes, 1, 1, padding='VALID', name='conv9', w_init=hk.initializers.UniformScaling((1/3)**0.5), b_init=hk.initializers.RandomUniform(-1/192**0.5, 1/192**0.5)
        )
        self.bn9 = hk.BatchNorm(True, True, 0.9, name='batchnorm9')
        self.avg_pool = hk.AvgPool(6, 6, 'VALID', name='avg_pool')
        
    def __call__(self, x, is_training):
        x = self.conv1(x)
        x = self.bn1(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv4(x)
        x = self.bn4(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv5(x)
        x = self.bn5(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv6(x)
        x = self.bn6(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv7(x)
        x = self.bn7(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv8(x)
        x = self.bn8(x, is_training)
        x = jax.nn.relu(x)
        x = self.conv9(x)
        x = self.bn9(x, is_training)
        x = jax.nn.relu(x)
        x = self.avg_pool(x)
        x = jnp.reshape(x, [*x.shape[:-3], -1])
        return x

if __name__ == "__main__":
    import optax
    import tree
    def forward(x, is_training):
        model = AllConv(10)
        return model(x, is_training)
    model_fn_t = hk.transform_with_state(forward)
    rng = jax.random.PRNGKey(44)
    M = 10
    keys = jax.random.split(rng, 10)
    parallel_init_fn = jax.vmap(model_fn_t.init, (0, None, None), 0)
    parallel_apply_fn = jax.vmap(model_fn_t.apply, (0, 0, None, None, None), 0)
    params, state = parallel_init_fn(keys, jnp.ones((1, 32, 32, 3)), True)
    print(params['AllConv/~/conv5']['w'].shape)
    # print(state)
    key, subkey = jax.random.split(rng)
    dummy_inputs = jax.random.normal(subkey, (4, 32, 32, 3))
    dummy_labels = jax.random.randint(subkey, (4,), 0, 10)
    print(dummy_labels)
    outputs, state = parallel_apply_fn(params, state, None, dummy_inputs, True)
    print(jax.nn.log_softmax(outputs, axis=-1))
    print(optax.softmax_cross_entropy_with_integer_labels(outputs, dummy_labels[None, :]))
    print(outputs.shape)
