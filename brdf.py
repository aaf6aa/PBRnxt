# Microfacet GGX specular model:
# Walter, B., Marschner, S.R., Li, H. and Torrance, K.E., 2007, June. Microfacet models for refraction through rough surfaces. In Proceedings of the 18th Eurographics conference on Rendering Techniques (pp. 195-206).
# https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf

# Disney Diffuse model:
# Lagarde, S. and De Rousiers, C., 2014. Moving frostbite to physically based rendering 3.0. SIGGRAPH Course: Physically Based Shading in Theory and Practice, 3.
# https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf

import numpy as np

import torch
import torch.nn as nn

def lerp(a, b, t):
    return a + t * (b - a)

def get_plane(b, h, w, plane_size, displacement, device=torch.device("cuda")):
    # Generate grid of points on the plane
    # Assume the plane is centered at the origin
    y_vals = torch.linspace(-plane_size/2, plane_size/2, h, device=device)
    x_vals = torch.linspace(-plane_size/2, plane_size/2, w, device=device)
    y_vals, x_vals = torch.meshgrid(y_vals, x_vals)
    
    # Plane coordinates with displacement in z
    plane = torch.zeros((b, h, w, 3), dtype=torch.float32, device=device)
    plane[..., 0] = x_vals
    plane[..., 1] = y_vals
    plane[..., 2] = displacement[...,0]

    return plane

def get_directions(plane, target_pos):
    directions = target_pos[None, None, None, :] - plane
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    return directions

def ray_march_occlusion(plane, direction):
    # ray march to find occlusion
    # for each point on the plane, march a ray towards the light source
    # if the ray hits the plane, then the point is in shadow
    # if the ray does not hit the plane, then the point is in light

    # plane: b x h x w x 3
    # direction: b x h x w x 3
    # returns: b x h x w x 1

    b, h, w, _ = plane.size()
    plane = plane.view(b * h * w, 3)
    direction = direction.view(b * h * w, 3)

    # plane normal
    plane_normal = torch.tensor([0.0, 0.0, 1.0], device=plane.device)

    # plane distance from origin
    plane_d = 0.0

    # ray-plane intersection
    t = - (torch.sum(plane * plane_normal, dim=-1) + plane_d) / torch.sum(direction * plane_normal, dim=-1)
    intersection = plane + t.view(b * h * w, 1) * direction

    # check if intersection is within the plane bounds
    intersection = intersection - plane
    intersection = intersection.view(b, h, w, 3)
    intersection = torch.norm(intersection, dim=-1)

    intersection = intersection.unsqueeze(-1).float()
    return intersection

MIN_DIELECTRICS_F0 = 0.04
def specular_f0(albedo, metalness):
    min_dielectric = torch.full((albedo.size(0), albedo.size(1), albedo.size(2), 1), MIN_DIELECTRICS_F0, dtype=torch.float32, device=albedo.device)
    return lerp(min_dielectric, albedo, metalness)

def luminance(color):
    return torch.sum(color * torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32, device=color.device), dim=-1, keepdim=True)

def eval_fresnel(f0, f90, l_dot_h):
    return f0 + (f90 - f0) * torch.pow(1.0 - l_dot_h, 5.0)

def ggx_d(alpha_sq, n_dot_h):
    b = ((alpha_sq - 1.0) * n_dot_h * n_dot_h + 1.0)
    return alpha_sq / (torch.pi * b * b)

def smith_g1(alpha, n_dot_s):
    alpha_sq = alpha * alpha
    n_dot_s_sq = n_dot_s * n_dot_s
    
    return 2.0 / (torch.sqrt(((alpha_sq * (1.0 - n_dot_s_sq)) + n_dot_s_sq) / n_dot_s_sq) + 1.0)

def smith_g2_ggx(alpha_sq, n_dot_l, n_dot_v):
    a = n_dot_v * torch.sqrt(alpha_sq + n_dot_l * (n_dot_l - alpha_sq * n_dot_l))
    b = n_dot_l * torch.sqrt(alpha_sq + n_dot_v * (n_dot_v - alpha_sq * n_dot_v))
    return 0.5 / (a + b)

def eval_brdf(n, l, v, albedo, roughness, metalness):
    # get normalized half vector
    h = (l + v)
    h = h / torch.norm(h, dim=-1, keepdim=True)

    n_dot_l = torch.clamp(torch.sum(n * l, dim=-1, keepdim=True), 0.00001, 1.0)
    n_dot_v = torch.clamp(torch.sum(n * v, dim=-1, keepdim=True), 0.00001, 1.0)
    
    l_dot_h = torch.clamp(torch.sum(l * h, dim=-1, keepdim=True), 0.00001, 1.0)
    n_dot_h = torch.clamp(torch.sum(n * h, dim=-1, keepdim=True), 0.00001, 1.0)

    f0 = specular_f0(albedo, metalness)
    reflectance = albedo * (1.0 - metalness)

    alpha_sq = roughness * roughness * roughness * roughness

    f90 = torch.clamp((1.0 / MIN_DIELECTRICS_F0) * luminance(f0), None, 1.0)

    f = eval_fresnel(f0, f90, l_dot_h)

    # specular
    d = ggx_d(torch.clamp(alpha_sq, 0.00001, None), n_dot_h)
    g2 = smith_g2_ggx(alpha_sq, n_dot_l, n_dot_v)

    specular = f * g2 * d * n_dot_l

    # diffuse
    energy_bias = 0.5 * roughness
    energy_factor = lerp(1.0, 1.0 / 1.51, roughness)

    fd90 = energy_bias + 2.0 * l_dot_h * l_dot_h * roughness - 1.0
    fdl = 1.0 + (fd90 * torch.pow(1.0 - n_dot_l, 5.0))
    fdv = 1.0 + (fd90 * torch.pow(1.0 - n_dot_v, 5.0))
    diffuse = fdl * fdv * energy_factor

    diffuse = reflectance * diffuse * (1.0 / np.pi) * n_dot_l

    return specular + diffuse

class Renderer(nn.Module):
    def __init__(self, device=torch.device('cuda')):
        super(Renderer, self).__init__()
        self.plane_size = 8
        
        self.plane_normal = torch.tensor([0.0, 0.0, 1.0], device=device)
        self.camera_pos = torch.tensor([ 0.0,  0.0, 5.0], device=device)
        self.light1_pos = torch.tensor([-4.0, -4.0, 10.0], device=device)
        self.light2_pos = torch.tensor([ 0.0,  0.0, 10.0], device=device)
        self.light3_pos = torch.tensor([ 4.0,  4.0, 10.0], device=device)
        self.light1_color = torch.tensor([1.0, 0.0, 0.0], device=device)
        self.light2_color = torch.tensor([0.0, 1.0, 0.0], device=device)
        self.light3_color = torch.tensor([0.0, 0.0, 1.0], device=device)
        self.def_roughness = torch.tensor([0.5], device=device)
        self.def_metalness = torch.tensor([0.5], device=device)
        self.def_displacement = torch.tensor([0.0], device=device)

    def step(self):
        self.light1_pos = torch.rand_like(self.light1_pos) * self.plane_size - self.plane_size / 2
        self.light1_pos[2] += self.plane_size / 2 + 2
        self.light2_pos = torch.rand_like(self.light2_pos) * self.plane_size - self.plane_size / 2
        self.light2_pos[2] += self.plane_size / 2 + 2
        self.light3_pos = torch.rand_like(self.light3_pos) * self.plane_size - self.plane_size / 2
        self.light3_pos[2] += self.plane_size / 2 + 2

        self.light1_color = torch.rand_like(self.light1_color)
        self.light2_color = torch.rand_like(self.light2_color)
        self.light3_color = torch.rand_like(self.light3_color)

        self.def_roughness = torch.rand_like(self.def_roughness).clamp(0.01, 0.99)
        self.def_metalness = torch.rand_like(self.def_metalness).clamp(0.01, 0.95)

    def render(self, albedo, normal, roughness=None, displacement=None, metalness=None):
        # bchw to bhwc
        b, c, h, w = albedo.shape
        if roughness is None:
            roughness = torch.ones((b, 1, h, w), device=albedo.device) * self.def_roughness
        if displacement is None:
            displacement = torch.ones((b, 1, h, w), device=albedo.device) * self.def_displacement
        if metalness is None:
            metalness = torch.ones((b, 1, h, w), device=albedo.device) * self.def_metalness

        # bchw to bhwc
        albedo = albedo.permute(0, 2, 3, 1).float()
        normal = normal.permute(0, 2, 3, 1).float()
        roughness = roughness.permute(0, 2, 3, 1).float()
        displacement = displacement.permute(0, 2, 3, 1).float()
        metalness = metalness.permute(0, 2, 3, 1).float()

        # add normal map to the default normal perpendicular to the surface
        normal = self.plane_normal + (normal * 2.0 - 1.0)
        normal = normal / torch.norm(normal, dim=-1, keepdim=True)

        plane = get_plane(b, albedo.size(1), albedo.size(2), self.plane_size, displacement, device=albedo.device)
        view_directions = get_directions(plane, self.camera_pos)
        light1_directions = get_directions(plane, self.light1_pos)
        light2_directions = get_directions(plane, self.light2_pos)
        light3_directions = get_directions(plane, self.light3_pos)

        occlusion1 = ray_march_occlusion(plane, light1_directions)
        occlusion2 = ray_march_occlusion(plane, light2_directions)
        occlusion3 = ray_march_occlusion(plane, light3_directions)
        
        light1 = eval_brdf(normal, light1_directions, view_directions, albedo, roughness, metalness)
        light2 = eval_brdf(normal, light2_directions, view_directions, albedo, roughness, metalness)
        light3 = eval_brdf(normal, light3_directions, view_directions, albedo, roughness, metalness)

        # adjust light intensity based on the angle between the normal and the light direction
        # grazing angles should have less intensity and facing angles should have more intensity
        cos_theta1 = torch.clamp(torch.sum(normal * light1_directions, dim=-1), 0.0, 1.0).unsqueeze(-1)
        cos_theta2 = torch.clamp(torch.sum(normal * light2_directions, dim=-1), 0.0, 1.0).unsqueeze(-1)
        cos_theta3 = torch.clamp(torch.sum(normal * light3_directions, dim=-1), 0.0, 1.0).unsqueeze(-1)
        
        rendered = (light1 * occlusion1 * self.light1_color * cos_theta1) + (light2 * occlusion2 * self.light2_color * cos_theta2) + (light3 * occlusion3 * self.light3_color * cos_theta3)
        rendered *= 6.0 # boost intensity for easier visualization

        # bhwc to bchw
        rendered = rendered.permute(0, 3, 1, 2)
        return rendered

