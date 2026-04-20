#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace
{
constexpr int kWindowWidth = 1400;
constexpr int kWindowHeight = 900;
constexpr float kAspect = static_cast<float>(kWindowWidth) / static_cast<float>(kWindowHeight);

constexpr float kG = 0.22f;
constexpr float kSoftening = 0.006f;
constexpr float kC = 150.0f; // Simulation speed-of-light scale for relativistic correction.
constexpr float kTimeStep = 0.0026f;
constexpr size_t kTrailLength = 320;

constexpr float kGridMin = -64.0f;
constexpr float kGridMax = 64.0f;
constexpr float kGridStep = 2.0f;
constexpr float kWarpScale = 120.0f;
constexpr float kWarpClamp = -5.5f;

struct Body
{
    glm::vec3 position{};
    glm::vec3 velocity{};
    glm::vec3 color{1.0f, 1.0f, 1.0f};
    float mass = 1.0f;
    float radius = 0.2f;
    std::string name{"UNNAMED"};
    GLuint textureId = 0;
    bool isBlackHole = false;
    bool isMoon = false;
    std::string parentName{};
    float moonOrbitRadius = 0.0f;
    float moonOrbitPhase = 0.0f;
    float moonOrbitRate = 0.0f;
    float axialTiltDeg = 0.0f;
    float spinRate = 0.0f; // radians per simulation second
    float spinAngle = 0.0f;
    std::deque<glm::vec3> trail{};
};

std::vector<Body> gBodies;
std::mt19937 gRng{std::random_device{}()};

GLuint gProgram = 0;
GLuint gSphereVao = 0;
GLuint gSphereVbo = 0;
GLuint gSphereEbo = 0;
GLsizei gSphereIndexCount = 0;

GLuint gGridVao = 0;
GLuint gGridVbo = 0;
GLsizei gGridVertexCount = 0;
GLuint gRingVao = 0;
GLuint gRingVbo = 0;
GLuint gRingEbo = 0;
GLsizei gRingIndexCount = 0;

glm::mat4 gView{};
glm::mat4 gProjection{};
glm::vec3 gLightPos{24.0f, 36.0f, 22.0f};

std::unordered_map<std::string, GLuint> gTextureCache;
GLuint gFallbackTexture = 0;
std::vector<GLuint> gDynamicTextures;

glm::vec3 gCameraPos{0.0f, 24.0f, 72.0f};
glm::vec3 gCameraTarget{0.0f, 0.0f, 0.0f};
glm::vec3 gCameraUp{0.0f, 1.0f, 0.0f};
float gCameraYaw = -90.0f;
float gCameraPitch = -22.0f;
float gCameraDistance = 75.0f;
bool gLeftMousePressed = false;
bool gRightMousePressed = false;
double gLastMouseX = 0.0;
double gLastMouseY = 0.0;
float gFov = 36.0f;

bool gSpawnPlanetPressed = false;
bool gSpawnBlackHolePressed = false;
bool gPausePressed = false;
bool gPaused = false;
int gSpawnedPlanetCount = 0;
int gSpawnedBlackHoleCount = 0;

const char* kVertexShader = R"(
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec2 aTexCoord;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProjection;

out vec3 vNormal;
out vec3 vWorldPos;
out vec2 vTexCoord;

void main()
{
    mat3 normalMat = mat3(transpose(inverse(uModel)));
    vNormal = normalize(normalMat * aNormal);
    vec4 worldPos = uModel * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vTexCoord = aTexCoord;
    gl_Position = uProjection * uView * worldPos;
}
)";

const char* kFragmentShader = R"(
#version 330 core
in vec3 vNormal;
in vec3 vWorldPos;
in vec2 vTexCoord;

uniform vec3 uColor;
uniform vec3 uLightPos;
uniform vec3 uViewPos;
uniform float uEmissive;
uniform sampler2D uTex;
uniform float uUseTexture;
uniform float uAlpha;

out vec4 FragColor;

void main()
{
    vec3 n = normalize(vNormal);
    vec3 lightDir = normalize(uLightPos - vWorldPos);
    vec3 viewDir = normalize(uViewPos - vWorldPos);
    vec3 halfway = normalize(lightDir + viewDir);

    float diff = max(dot(n, lightDir), 0.0);
    float spec = pow(max(dot(n, halfway), 0.0), 48.0);
    float ambient = 0.08;

    vec4 texel = texture(uTex, vTexCoord);
    vec3 texColor = texel.rgb;
    vec3 baseColor = mix(uColor, texColor, uUseTexture);
    vec3 lit = baseColor * (ambient + 0.95 * diff) + vec3(0.9) * spec * 0.4;
    lit += uColor * uEmissive;
    float alpha = mix(1.0, texel.a, uUseTexture) * uAlpha;
    FragColor = vec4(lit, alpha);
}
)";

GLuint compileShader(GLenum type, const char* src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
        std::string msg(static_cast<size_t>(len), '\0');
        glGetShaderInfoLog(shader, len, nullptr, msg.data());
        std::cerr << "Shader compile error:\n" << msg << "\n";
    }
    return shader;
}

GLuint makeProgram()
{
    GLuint vs = compileShader(GL_VERTEX_SHADER, kVertexShader);
    GLuint fs = compileShader(GL_FRAGMENT_SHADER, kFragmentShader);
    GLuint program = glCreateProgram();
    glAttachShader(program, vs);
    glAttachShader(program, fs);
    glLinkProgram(program);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok)
    {
        GLint len = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &len);
        std::string msg(static_cast<size_t>(len), '\0');
        glGetProgramInfoLog(program, len, nullptr, msg.data());
        std::cerr << "Program link error:\n" << msg << "\n";
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
    return program;
}

size_t addBody(const Body& b)
{
    gBodies.push_back(b);
    if (gBodies.back().textureId == 0)
    {
        gBodies.back().textureId = gFallbackTexture;
    }
    gBodies.back().trail.push_back(b.position);
    return gBodies.size() - 1;
}

float hashNoise(float x, float y, float z)
{
    float n = std::sin(x * 12.9898f + y * 78.233f + z * 37.719f) * 43758.5453f;
    return n - std::floor(n);
}

GLuint createTextureFromData(int w, int h, const std::vector<unsigned char>& data)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D);
    return tex;
}

GLuint createTextureFromDataRGBA(int w, int h, const std::vector<unsigned char>& data)
{
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glGenerateMipmap(GL_TEXTURE_2D);
    return tex;
}

GLuint makePlanetTexture(const glm::vec3& baseColor, float stripeStrength, float roughness, bool hasIceCap)
{
    constexpr int kTexSize = 512;
    std::vector<unsigned char> pixels(static_cast<size_t>(kTexSize) * kTexSize * 3);

    for (int y = 0; y < kTexSize; ++y)
    {
        float v = static_cast<float>(y) / static_cast<float>(kTexSize - 1);
        float lat = (v - 0.5f) * glm::pi<float>();
        for (int x = 0; x < kTexSize; ++x)
        {
            float u = static_cast<float>(x) / static_cast<float>(kTexSize - 1);
            float lon = u * glm::two_pi<float>();
            float stripes = std::sin(lat * 22.0f + hashNoise(u * 3.0f, v * 7.0f, 1.0f) * 4.0f) * stripeStrength;
            float crater = (hashNoise(u * 28.0f, v * 36.0f, 2.0f) - 0.5f) * roughness;
            float cloud = (hashNoise(u * 10.0f, v * 10.0f, 9.0f) - 0.5f) * 0.12f;
            float polar = hasIceCap ? glm::smoothstep(0.72f, 0.95f, std::abs(std::sin(lat))) : 0.0f;

            glm::vec3 color = baseColor;
            color += glm::vec3(stripes + crater + cloud);
            color = glm::mix(color, glm::vec3(0.93f, 0.95f, 1.0f), polar * 0.85f);
            color = glm::clamp(color, glm::vec3(0.02f), glm::vec3(1.0f));

            size_t idx = static_cast<size_t>(y * kTexSize + x) * 3;
            pixels[idx + 0] = static_cast<unsigned char>(color.r * 255.0f);
            pixels[idx + 1] = static_cast<unsigned char>(color.g * 255.0f);
            pixels[idx + 2] = static_cast<unsigned char>(color.b * 255.0f);
            (void)lon;
        }
    }

    return createTextureFromData(kTexSize, kTexSize, pixels);
}

GLuint makeSunTexture()
{
    constexpr int kTexSize = 512;
    std::vector<unsigned char> pixels(static_cast<size_t>(kTexSize) * kTexSize * 3);
    for (int y = 0; y < kTexSize; ++y)
    {
        for (int x = 0; x < kTexSize; ++x)
        {
            float u = static_cast<float>(x) / static_cast<float>(kTexSize - 1);
            float v = static_cast<float>(y) / static_cast<float>(kTexSize - 1);
            float turbulence = hashNoise(u * 12.0f, v * 12.0f, 4.0f) * 0.45f;
            glm::vec3 c = glm::vec3(1.0f, 0.65f, 0.10f) + glm::vec3(turbulence, turbulence * 0.6f, 0.0f);
            c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
            size_t idx = static_cast<size_t>(y * kTexSize + x) * 3;
            pixels[idx + 0] = static_cast<unsigned char>(c.r * 255.0f);
            pixels[idx + 1] = static_cast<unsigned char>(c.g * 255.0f);
            pixels[idx + 2] = static_cast<unsigned char>(c.b * 255.0f);
        }
    }
    return createTextureFromData(kTexSize, kTexSize, pixels);
}

GLuint makeSaturnRingTexture()
{
    constexpr int kW = 2048;
    constexpr int kH = 256;
    std::vector<unsigned char> pixels(static_cast<size_t>(kW) * kH * 4);
    for (int y = 0; y < kH; ++y)
    {
        float v = static_cast<float>(y) / static_cast<float>(kH - 1); // radial coordinate
        float edgeFade = glm::smoothstep(0.02f, 0.12f, v) * (1.0f - glm::smoothstep(0.88f, 0.98f, v));
        for (int x = 0; x < kW; ++x)
        {
            float u = static_cast<float>(x) / static_cast<float>(kW - 1);
            float radialBands = 0.5f + 0.5f * std::sin(v * 210.0f + hashNoise(v * 45.0f, u * 4.0f, 3.0f) * 6.0f);
            float azimuthNoise = (hashNoise(u * 75.0f, v * 18.0f, 9.0f) - 0.5f) * 0.15f;
            float ringGap = glm::smoothstep(0.74f, 0.79f, std::abs(std::sin(v * 38.0f + 0.6f)));
            float alpha = edgeFade * (0.38f + radialBands * 0.44f) * (1.0f - 0.45f * ringGap);
            alpha *= 0.85f + azimuthNoise;
            alpha = glm::clamp(alpha, 0.0f, 0.95f);

            float brightness = 0.55f + radialBands * 0.38f + azimuthNoise;
            glm::vec3 color = glm::vec3(0.78f, 0.71f, 0.61f) * brightness;
            color = glm::clamp(color, glm::vec3(0.22f), glm::vec3(0.97f));

            size_t idx = static_cast<size_t>(y * kW + x) * 4;
            pixels[idx + 0] = static_cast<unsigned char>(color.r * 255.0f);
            pixels[idx + 1] = static_cast<unsigned char>(color.g * 255.0f);
            pixels[idx + 2] = static_cast<unsigned char>(color.b * 255.0f);
            pixels[idx + 3] = static_cast<unsigned char>(alpha * 255.0f);
        }
    }
    return createTextureFromDataRGBA(kW, kH, pixels);
}

void initializeTextures()
{
    gFallbackTexture = makePlanetTexture({0.6f, 0.6f, 0.6f}, 0.0f, 0.15f, false);
    gTextureCache["SUN"] = makeSunTexture();
    gTextureCache["MERCURY"] = makePlanetTexture({0.62f, 0.63f, 0.65f}, 0.10f, 0.20f, false);
    gTextureCache["VENUS"] = makePlanetTexture({0.85f, 0.74f, 0.40f}, 0.15f, 0.08f, false);
    gTextureCache["EARTH"] = makePlanetTexture({0.18f, 0.45f, 0.82f}, 0.08f, 0.22f, true);
    gTextureCache["MOON"] = makePlanetTexture({0.72f, 0.72f, 0.74f}, 0.03f, 0.24f, false);
    gTextureCache["MARS"] = makePlanetTexture({0.80f, 0.34f, 0.24f}, 0.08f, 0.18f, false);
    gTextureCache["JUPITER"] = makePlanetTexture({0.86f, 0.67f, 0.52f}, 0.42f, 0.04f, false);
    gTextureCache["SATURN"] = makePlanetTexture({0.88f, 0.78f, 0.58f}, 0.32f, 0.04f, false);
    gTextureCache["URANUS"] = makePlanetTexture({0.56f, 0.86f, 0.90f}, 0.06f, 0.06f, false);
    gTextureCache["NEPTUNE"] = makePlanetTexture({0.28f, 0.43f, 0.90f}, 0.09f, 0.06f, false);
    gTextureCache["PLUTO"] = makePlanetTexture({0.72f, 0.67f, 0.60f}, 0.06f, 0.16f, false);
    gTextureCache["BLACK HOLE"] = makePlanetTexture({0.05f, 0.05f, 0.07f}, 0.0f, 0.0f, false);
    gTextureCache["SATURN_RING"] = makeSaturnRingTexture();
}

GLuint textureForBodyName(const std::string& bodyName)
{
    for (const auto& [key, tex] : gTextureCache)
    {
        if (bodyName.find(key) != std::string::npos)
        {
            return tex;
        }
    }
    return gFallbackTexture;
}

void assignBodyTexture(Body& body)
{
    body.textureId = textureForBodyName(body.name);
}

void applySpinDefaults(Body& body)
{
    if (body.name == "SUN") { body.spinRate = 0.18f; body.axialTiltDeg = 7.0f; return; }
    if (body.name == "MERCURY") { body.spinRate = 0.05f; body.axialTiltDeg = 0.1f; return; }
    if (body.name == "VENUS") { body.spinRate = -0.03f; body.axialTiltDeg = 177.0f; return; }
    if (body.name == "EARTH") { body.spinRate = 0.85f; body.axialTiltDeg = 23.4f; return; }
    if (body.name == "MOON") { body.spinRate = 0.08f; body.axialTiltDeg = 6.7f; return; }
    if (body.name == "MARS") { body.spinRate = 0.78f; body.axialTiltDeg = 25.2f; return; }
    if (body.name == "JUPITER") { body.spinRate = 1.15f; body.axialTiltDeg = 3.1f; return; }
    if (body.name == "SATURN") { body.spinRate = 0.95f; body.axialTiltDeg = 26.7f; return; }
    if (body.name == "URANUS") { body.spinRate = 0.52f; body.axialTiltDeg = 97.8f; return; }
    if (body.name == "NEPTUNE") { body.spinRate = 0.72f; body.axialTiltDeg = 28.3f; return; }
    if (body.name == "PLUTO") { body.spinRate = 0.11f; body.axialTiltDeg = 122.5f; return; }
    if (body.name.find("BLACK HOLE") != std::string::npos) { body.spinRate = 0.42f; body.axialTiltDeg = 12.0f; return; }
    body.spinRate = 0.45f;
    body.axialTiltDeg = 8.0f;
}

void updateCameraView()
{
    gCameraPitch = std::clamp(gCameraPitch, -88.0f, 88.0f);
    gCameraDistance = std::clamp(gCameraDistance, 10.0f, 220.0f);
    const float yawRad = glm::radians(gCameraYaw);
    const float pitchRad = glm::radians(gCameraPitch);
    const glm::vec3 direction{
        std::cos(pitchRad) * std::cos(yawRad),
        std::sin(pitchRad),
        std::cos(pitchRad) * std::sin(yawRad)};
    gCameraPos = gCameraTarget - direction * gCameraDistance;
    gView = glm::lookAt(gCameraPos, gCameraTarget, gCameraUp);
}

void resetCamera()
{
    gCameraYaw = -90.0f;
    gCameraPitch = -22.0f;
    gCameraDistance = 75.0f;
    gCameraTarget = glm::vec3(0.0f);
    gFov = 36.0f;
    updateCameraView();
}

using Glyph = std::array<uint8_t, 7>;

Glyph glyphFor(char c)
{
    switch (c)
    {
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    case 'D': return {0x1E, 0x12, 0x11, 0x11, 0x11, 0x12, 0x1E};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'G': return {0x0E, 0x11, 0x10, 0x10, 0x13, 0x11, 0x0E};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F};
    case 'J': return {0x01, 0x01, 0x01, 0x01, 0x11, 0x11, 0x0E};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    case 'M': return {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11};
    case 'O': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'P': return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    case 'Q': return {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    case 'X': return {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    case 'Y': return {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    case '0': return {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    case '1': return {0x04, 0x0C, 0x14, 0x04, 0x04, 0x04, 0x1F};
    case '2': return {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    case '3': return {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    case '4': return {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    case '5': return {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    case '6': return {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    case '9': return {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};
    case '#': return {0x0A, 0x0A, 0x1F, 0x0A, 0x1F, 0x0A, 0x0A};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    default: return {0x1F, 0x11, 0x01, 0x02, 0x04, 0x00, 0x04}; // '?'
    }
}

void drawText2D(const std::string& text, float x, float y, float scale, int w, int h)
{
    glUseProgram(0);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, static_cast<double>(w), 0.0, static_cast<double>(h), -1.0, 1.0);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glColor3f(1.0f, 1.0f, 1.0f);

    float cx = x;
    for (char raw : text)
    {
        char c = static_cast<char>(std::toupper(static_cast<unsigned char>(raw)));
        const Glyph glyph = glyphFor(c);
        for (int row = 0; row < 7; ++row)
        {
            for (int col = 0; col < 5; ++col)
            {
                if (((glyph[row] >> (4 - col)) & 0x1) == 0)
                {
                    continue;
                }
                float px = cx + static_cast<float>(col) * scale;
                float py = y - static_cast<float>(row) * scale;
                glBegin(GL_QUADS);
                glVertex2f(px, py);
                glVertex2f(px + scale, py);
                glVertex2f(px + scale, py - scale);
                glVertex2f(px, py - scale);
                glEnd();
            }
        }
        cx += 6.0f * scale;
    }

    glEnable(GL_DEPTH_TEST);
    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawBodyLabels(GLFWwindow* window)
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    const glm::vec4 viewport(0.0f, 0.0f, static_cast<float>(width), static_cast<float>(height));

    for (const auto& body : gBodies)
    {
        glm::vec3 anchor = body.position + glm::vec3(0.0f, body.radius + 0.85f, 0.0f);
        glm::vec3 screen = glm::project(anchor, gView, gProjection, viewport);
        if (screen.z < 0.0f || screen.z > 1.0f)
        {
            continue;
        }
        const float scale = 1.8f;
        const float textWidth = static_cast<float>(body.name.size()) * 6.0f * scale;
        drawText2D(body.name, screen.x - textWidth * 0.5f, screen.y, scale, width, height);
    }
}

void seedSolarSystem()
{
    Body sun;
    sun.position = {0.0f, 0.0f, 0.0f};
    sun.velocity = {0.0f, 0.0f, 0.0f};
    sun.mass = 12000.0f;
    sun.radius = 2.0f;
    sun.name = "SUN";
    sun.color = {1.0f, 0.77f, 0.18f};
    assignBodyTexture(sun);
    applySpinDefaults(sun);
    addBody(sun);

    auto makeOrbitPlanet = [&](const std::string& name, float orbitalRadius, float mass, float radius, const glm::vec3& color, float speedFactor = 1.0f) {
        Body p;
        p.position = {orbitalRadius, 0.0f, 0.0f};
        float speed = std::sqrt(kG * sun.mass / orbitalRadius) * speedFactor;
        p.velocity = {0.0f, 0.0f, speed};
        p.mass = mass;
        p.radius = radius;
        p.name = name;
        p.color = color;
        assignBodyTexture(p);
        applySpinDefaults(p);
        return addBody(p);
    };

    makeOrbitPlanet("MERCURY", 6.0f, 0.30f, 0.20f, {0.7f, 0.75f, 0.85f}, 1.01f);
    makeOrbitPlanet("VENUS", 8.1f, 0.82f, 0.28f, {0.95f, 0.82f, 0.4f}, 1.0f);
    const size_t earthIdx = makeOrbitPlanet("EARTH", 10.3f, 1.0f, 0.30f, {0.35f, 0.6f, 1.0f}, 1.0f);
    makeOrbitPlanet("MARS", 12.6f, 0.55f, 0.23f, {1.0f, 0.45f, 0.32f}, 1.0f);
    makeOrbitPlanet("JUPITER", 17.6f, 36.0f, 0.86f, {0.95f, 0.7f, 0.5f}, 1.0f);
    makeOrbitPlanet("SATURN", 23.5f, 18.0f, 0.74f, {0.9f, 0.8f, 0.62f}, 1.0f);
    makeOrbitPlanet("URANUS", 29.8f, 6.6f, 0.58f, {0.55f, 0.88f, 0.9f}, 1.0f);
    makeOrbitPlanet("NEPTUNE", 35.7f, 7.6f, 0.56f, {0.32f, 0.46f, 0.95f}, 1.0f);
    makeOrbitPlanet("PLUTO", 41.2f, 0.09f, 0.16f, {0.78f, 0.7f, 0.62f}, 1.0f);

    Body moon;
    moon.name = "MOON";
    moon.mass = 0.0085f;
    moon.radius = 0.11f;
    moon.color = {0.85f, 0.85f, 0.9f};
    moon.isMoon = true;
    moon.parentName = "EARTH";
    moon.moonOrbitRadius = 0.92f;
    moon.moonOrbitPhase = 0.2f;
    moon.moonOrbitRate = std::sqrt(kG * gBodies[earthIdx].mass / (moon.moonOrbitRadius * moon.moonOrbitRadius * moon.moonOrbitRadius));
    moon.position = gBodies[earthIdx].position + glm::vec3(moon.moonOrbitRadius, 0.06f, 0.0f);
    moon.velocity = gBodies[earthIdx].velocity;
    assignBodyTexture(moon);
    applySpinDefaults(moon);
    addBody(moon);
}

void spawnPlanet()
{
    std::uniform_real_distribution<float> rDist(15.0f, 52.0f);
    std::uniform_real_distribution<float> angleDist(0.0f, glm::two_pi<float>());
    std::uniform_real_distribution<float> cDist(0.2f, 1.0f);
    std::uniform_real_distribution<float> mDist(0.2f, 2.4f);
    std::uniform_real_distribution<float> radDist(0.18f, 0.44f);

    float r = rDist(gRng);
    float angle = angleDist(gRng);
    glm::vec3 pos{r * std::cos(angle), 0.0f, r * std::sin(angle)};

    const Body& center = gBodies.front();
    glm::vec3 dirToCenter = glm::normalize(center.position - pos);
    glm::vec3 tangent = glm::normalize(glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), dirToCenter));
    float speed = std::sqrt(kG * center.mass / r) * 0.96f;

    Body p;
    ++gSpawnedPlanetCount;
    p.name = "PLANET #" + std::to_string(gSpawnedPlanetCount);
    p.position = pos;
    p.velocity = tangent * speed;
    p.mass = mDist(gRng);
    p.radius = radDist(gRng);
    p.color = {cDist(gRng), cDist(gRng), cDist(gRng)};
    p.textureId = makePlanetTexture(p.color, 0.10f + cDist(gRng) * 0.26f, 0.06f + cDist(gRng) * 0.20f, cDist(gRng) > 0.74f);
    gDynamicTextures.push_back(p.textureId);
    applySpinDefaults(p);
    addBody(p);
}

void spawnBlackHole()
{
    std::uniform_real_distribution<float> angleDist(0.0f, glm::two_pi<float>());
    float r = 48.0f;
    float angle = angleDist(gRng);

    Body bh;
    ++gSpawnedBlackHoleCount;
    bh.name = "BLACK HOLE #" + std::to_string(gSpawnedBlackHoleCount);
    bh.position = {r * std::cos(angle), 0.0f, r * std::sin(angle)};
    bh.velocity = {0.0f, 0.0f, std::sqrt(kG * gBodies.front().mass / r) * 0.68f};
    bh.mass = 1300.0f;
    bh.radius = 0.82f;
    bh.color = {0.05f, 0.05f, 0.08f};
    bh.isBlackHole = true;
    assignBodyTexture(bh);
    applySpinDefaults(bh);
    addBody(bh);
}

void buildSphereMesh(int stacks, int slices)
{
    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;

    for (int i = 0; i <= stacks; ++i)
    {
        float v = static_cast<float>(i) / static_cast<float>(stacks);
        float phi = v * glm::pi<float>();
        for (int j = 0; j <= slices; ++j)
        {
            float u = static_cast<float>(j) / static_cast<float>(slices);
            float theta = u * glm::two_pi<float>();
            glm::vec3 p{
                std::sin(phi) * std::cos(theta),
                std::cos(phi),
                std::sin(phi) * std::sin(theta)};
            vertices.push_back({p, glm::normalize(p), {u, 1.0f - v}});
        }
    }

    for (int i = 0; i < stacks; ++i)
    {
        for (int j = 0; j < slices; ++j)
        {
            int row1 = i * (slices + 1);
            int row2 = (i + 1) * (slices + 1);

            indices.push_back(row1 + j);
            indices.push_back(row2 + j);
            indices.push_back(row1 + j + 1);

            indices.push_back(row1 + j + 1);
            indices.push_back(row2 + j);
            indices.push_back(row2 + j + 1);
        }
    }

    gSphereIndexCount = static_cast<GLsizei>(indices.size());

    glGenVertexArrays(1, &gSphereVao);
    glGenBuffers(1, &gSphereVbo);
    glGenBuffers(1, &gSphereEbo);

    glBindVertexArray(gSphereVao);
    glBindBuffer(GL_ARRAY_BUFFER, gSphereVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gSphereEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)), indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

void buildRingMesh(float innerRadius, float outerRadius, int segments)
{
    struct Vertex
    {
        glm::vec3 pos;
        glm::vec3 normal;
        glm::vec2 uv;
    };

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    vertices.reserve(static_cast<size_t>(segments + 1) * 2);
    indices.reserve(static_cast<size_t>(segments) * 6);

    for (int i = 0; i <= segments; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(segments);
        float angle = t * glm::two_pi<float>();
        float c = std::cos(angle);
        float s = std::sin(angle);

        vertices.push_back({glm::vec3(innerRadius * c, 0.0f, innerRadius * s), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(t, 0.0f)});
        vertices.push_back({glm::vec3(outerRadius * c, 0.0f, outerRadius * s), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec2(t, 1.0f)});
    }

    for (int i = 0; i < segments; ++i)
    {
        unsigned int a = static_cast<unsigned int>(i * 2);
        unsigned int b = a + 1;
        unsigned int c = a + 2;
        unsigned int d = a + 3;
        indices.push_back(a);
        indices.push_back(c);
        indices.push_back(b);
        indices.push_back(b);
        indices.push_back(c);
        indices.push_back(d);
    }

    gRingIndexCount = static_cast<GLsizei>(indices.size());
    glGenVertexArrays(1, &gRingVao);
    glGenBuffers(1, &gRingVbo);
    glGenBuffers(1, &gRingEbo);

    glBindVertexArray(gRingVao);
    glBindBuffer(GL_ARRAY_BUFFER, gRingVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(Vertex)), vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gRingEbo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(indices.size() * sizeof(unsigned int)), indices.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, normal)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void*>(offsetof(Vertex, uv)));
    glEnableVertexAttribArray(2);
    glBindVertexArray(0);
}

float computeWarpHeight(const glm::vec2& xz)
{
    // Physics-inspired curvature proxy:
    // base paraboloid + negative potential contribution from all masses.
    float paraboloid = 0.0025f * (xz.x * xz.x + xz.y * xz.y);
    float potential = 0.0f;
    for (const auto& b : gBodies)
    {
        glm::vec2 bp{b.position.x, b.position.z};
        float r = glm::length(xz - bp);
        potential += (kG * b.mass) / std::sqrt(r * r + 0.20f);
    }
    float warp = paraboloid - potential / kWarpScale;
    return std::max(warp, kWarpClamp);
}

void buildGrid()
{
    std::vector<float> vertices;
    auto pushLine = [&](glm::vec3 a, glm::vec3 b) {
        vertices.push_back(a.x);
        vertices.push_back(a.y);
        vertices.push_back(a.z);
        vertices.push_back(0.0f);
        vertices.push_back(1.0f);
        vertices.push_back(0.0f);

        vertices.push_back(b.x);
        vertices.push_back(b.y);
        vertices.push_back(b.z);
        vertices.push_back(0.0f);
        vertices.push_back(1.0f);
        vertices.push_back(0.0f);
    };

    for (float x = kGridMin; x <= kGridMax; x += kGridStep)
    {
        glm::vec3 prev{x, computeWarpHeight({x, kGridMin}), kGridMin};
        for (float z = kGridMin + kGridStep; z <= kGridMax; z += kGridStep)
        {
            glm::vec3 cur{x, computeWarpHeight({x, z}), z};
            pushLine(prev, cur);
            prev = cur;
        }
    }

    for (float z = kGridMin; z <= kGridMax; z += kGridStep)
    {
        glm::vec3 prev{kGridMin, computeWarpHeight({kGridMin, z}), z};
        for (float x = kGridMin + kGridStep; x <= kGridMax; x += kGridStep)
        {
            glm::vec3 cur{x, computeWarpHeight({x, z}), z};
            pushLine(prev, cur);
            prev = cur;
        }
    }

    gGridVertexCount = static_cast<GLsizei>(vertices.size() / 6);

    if (gGridVao == 0)
    {
        glGenVertexArrays(1, &gGridVao);
        glGenBuffers(1, &gGridVbo);
    }

    glBindVertexArray(gGridVao);
    glBindBuffer(GL_ARRAY_BUFFER, gGridVbo);
    glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(float)), vertices.data(), GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void resolveCollisions()
{
    for (size_t i = 0; i < gBodies.size(); ++i)
    {
        for (size_t j = i + 1; j < gBodies.size();)
        {
            float r = glm::length(gBodies[j].position - gBodies[i].position);
            float combined = gBodies[i].radius + gBodies[j].radius;
            if (r <= combined)
            {
                // Perfectly inelastic merge preserving momentum.
                Body merged;
                float m1 = gBodies[i].mass;
                float m2 = gBodies[j].mass;
                float m = m1 + m2;
                merged.mass = m;
                merged.position = (gBodies[i].position * m1 + gBodies[j].position * m2) / m;
                merged.velocity = (gBodies[i].velocity * m1 + gBodies[j].velocity * m2) / m;
                merged.radius = std::cbrt(std::pow(gBodies[i].radius, 3.0f) + std::pow(gBodies[j].radius, 3.0f));
                merged.color = (gBodies[i].color * m1 + gBodies[j].color * m2) / m;
                merged.isBlackHole = gBodies[i].isBlackHole || gBodies[j].isBlackHole;
                merged.name = (gBodies[i].mass >= gBodies[j].mass) ? gBodies[i].name : gBodies[j].name;
                merged.spinRate = (gBodies[i].spinRate * m1 + gBodies[j].spinRate * m2) / m;
                merged.spinAngle = (gBodies[i].spinAngle + gBodies[j].spinAngle) * 0.5f;
                merged.axialTiltDeg = (gBodies[i].axialTiltDeg * m1 + gBodies[j].axialTiltDeg * m2) / m;
                merged.isMoon = false;
                merged.parentName.clear();

                if (merged.isBlackHole)
                {
                    merged.color = {0.06f, 0.06f, 0.08f};
                    merged.radius *= 0.9f;
                    if (merged.name.find("BLACK HOLE") == std::string::npos)
                    {
                        merged.name = "BLACK HOLE MERGED";
                    }
                }
                assignBodyTexture(merged);

                merged.trail = gBodies[i].trail;
                if (merged.trail.size() < gBodies[j].trail.size())
                {
                    merged.trail = gBodies[j].trail;
                }
                merged.trail.push_back(merged.position);

                gBodies[i] = merged;
                gBodies.erase(gBodies.begin() + static_cast<long long>(j));
            }
            else
            {
                ++j;
            }
        }
    }
}

void integratePhysics()
{
    const size_t n = gBodies.size();
    std::vector<glm::vec3> accel(n, glm::vec3(0.0f));
    std::vector<bool> dynamicBody(n, true);
    for (size_t i = 0; i < n; ++i)
    {
        if (gBodies[i].isMoon && !gBodies[i].parentName.empty())
        {
            dynamicBody[i] = false;
        }
    }

    for (size_t i = 0; i < n; ++i)
    {
        if (!dynamicBody[i])
        {
            continue;
        }
        for (size_t j = i + 1; j < n; ++j)
        {
            if (!dynamicBody[j])
            {
                continue;
            }
            glm::vec3 rVec = gBodies[j].position - gBodies[i].position;
            float dist2 = glm::dot(rVec, rVec) + kSoftening;
            float dist = std::sqrt(dist2);
            glm::vec3 dir = rVec / dist;

            // Newtonian force.
            float a1Newton = kG * gBodies[j].mass / dist2;
            float a2Newton = kG * gBodies[i].mass / dist2;

            // Simple relativistic correction:
            // stronger effective gravity at high speed / deep potential.
            float v1 = glm::length(gBodies[i].velocity);
            float v2 = glm::length(gBodies[j].velocity);
            float gamma1 = 1.0f / std::sqrt(std::max(1.0f - (v1 * v1) / (kC * kC), 0.2f));
            float gamma2 = 1.0f / std::sqrt(std::max(1.0f - (v2 * v2) / (kC * kC), 0.2f));
            float grBoost = 1.0f + (3.0f * kG * (gBodies[i].mass + gBodies[j].mass)) / (dist * kC * kC);

            float a1 = a1Newton * gamma1 * grBoost;
            float a2 = a2Newton * gamma2 * grBoost;

            accel[i] += dir * a1;
            accel[j] -= dir * a2;
        }
    }

    for (size_t i = 0; i < n; ++i)
    {
        gBodies[i].spinAngle += gBodies[i].spinRate * kTimeStep;
        if (dynamicBody[i])
        {
            gBodies[i].velocity += accel[i] * kTimeStep;
            gBodies[i].position += gBodies[i].velocity * kTimeStep;
        }
    }

    // Keep moons in clear revolutions around their parent planet for visual stability.
    for (size_t i = 0; i < n; ++i)
    {
        if (!(gBodies[i].isMoon && !gBodies[i].parentName.empty()))
        {
            continue;
        }
        auto parentIt = std::find_if(gBodies.begin(), gBodies.end(), [&](const Body& b) { return b.name == gBodies[i].parentName; });
        if (parentIt == gBodies.end())
        {
            continue;
        }
        Body& moon = gBodies[i];
        const Body& parent = *parentIt;
        moon.moonOrbitPhase += moon.moonOrbitRate * kTimeStep;
        glm::vec3 offset{
            std::cos(moon.moonOrbitPhase) * moon.moonOrbitRadius,
            std::sin(moon.moonOrbitPhase * 0.55f) * (moon.moonOrbitRadius * 0.16f),
            std::sin(moon.moonOrbitPhase) * moon.moonOrbitRadius};
        moon.position = parent.position + offset;
        glm::vec3 tangent{
            -std::sin(moon.moonOrbitPhase),
            std::cos(moon.moonOrbitPhase * 0.55f) * 0.55f * 0.16f,
            std::cos(moon.moonOrbitPhase)};
        moon.velocity = parent.velocity + glm::normalize(tangent) * (moon.moonOrbitRadius * moon.moonOrbitRate);
    }

    for (size_t i = 0; i < n; ++i)
    {
        gBodies[i].trail.push_back(gBodies[i].position);
        if (gBodies[i].trail.size() > kTrailLength)
        {
            gBodies[i].trail.pop_front();
        }
    }

    resolveCollisions();
}

void drawBody(const Body& body, GLint modelLoc, GLint colorLoc, GLint emissiveLoc, GLint useTextureLoc, GLint alphaLoc)
{
    glm::mat4 model = glm::translate(glm::mat4(1.0f), body.position);
    model = glm::rotate(model, glm::radians(body.axialTiltDeg), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::rotate(model, body.spinAngle, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::scale(model, glm::vec3(body.radius));
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3fv(colorLoc, 1, glm::value_ptr(body.color));

    float emissive = 0.0f;
    if (body.mass > 1000.0f)
    {
        emissive = 0.30f;
    }
    if (body.isBlackHole)
    {
        emissive = 0.12f;
    }
    glUniform1f(emissiveLoc, emissive);
    glUniform1f(useTextureLoc, 1.0f);
    glUniform1f(alphaLoc, 1.0f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, body.textureId != 0 ? body.textureId : gFallbackTexture);

    glBindVertexArray(gSphereVao);
    glDrawElements(GL_TRIANGLES, gSphereIndexCount, GL_UNSIGNED_INT, nullptr);
}

void drawTrails(GLint modelLoc, GLint colorLoc, GLint emissiveLoc, GLint useTextureLoc, GLint alphaLoc)
{
    glUniform1f(emissiveLoc, 0.0f);
    glUniform1f(useTextureLoc, 0.0f);
    glUniform1f(alphaLoc, 1.0f);
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));

    for (const auto& b : gBodies)
    {
        if (b.trail.size() < 2)
        {
            continue;
        }
        glUniform3fv(colorLoc, 1, glm::value_ptr(glm::mix(b.color, glm::vec3(1.0f), 0.30f)));
        glBegin(GL_LINE_STRIP);
        for (const auto& p : b.trail)
        {
            glVertex3f(p.x, p.y, p.z);
        }
        glEnd();
    }
}

void drawGrid(GLint modelLoc, GLint colorLoc, GLint emissiveLoc, GLint useTextureLoc, GLint alphaLoc)
{
    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(glm::mat4(1.0f)));
    glUniform3f(colorLoc, 1.0f, 1.0f, 1.0f);
    glUniform1f(emissiveLoc, 0.55f);
    glUniform1f(useTextureLoc, 0.0f);
    glUniform1f(alphaLoc, 1.0f);
    glBindVertexArray(gGridVao);
    glDrawArrays(GL_LINES, 0, gGridVertexCount);
}

void drawSaturnRing(const Body& saturn, GLint modelLoc, GLint colorLoc, GLint emissiveLoc, GLint useTextureLoc, GLint alphaLoc)
{
    if (gRingVao == 0 || gRingIndexCount == 0)
    {
        return;
    }
    auto it = gTextureCache.find("SATURN_RING");
    if (it == gTextureCache.end())
    {
        return;
    }

    glm::mat4 model = glm::translate(glm::mat4(1.0f), saturn.position);
    model = glm::rotate(model, glm::radians(saturn.axialTiltDeg), glm::vec3(0.0f, 0.0f, 1.0f));
    model = glm::rotate(model, saturn.spinAngle, glm::vec3(0.0f, 1.0f, 0.0f));
    model = glm::scale(model, glm::vec3(saturn.radius * 2.35f));

    glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
    glUniform3f(colorLoc, 0.85f, 0.78f, 0.66f);
    glUniform1f(emissiveLoc, 0.02f);
    glUniform1f(useTextureLoc, 1.0f);
    glUniform1f(alphaLoc, 0.94f);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, it->second);

    glBindVertexArray(gRingVao);
    glDrawElements(GL_TRIANGLES, gRingIndexCount, GL_UNSIGNED_INT, nullptr);
}

void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
    {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    }

    bool pDown = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS;
    if (pDown && !gSpawnPlanetPressed)
    {
        spawnPlanet();
    }
    gSpawnPlanetPressed = pDown;

    bool bDown = glfwGetKey(window, GLFW_KEY_B) == GLFW_PRESS;
    if (bDown && !gSpawnBlackHolePressed)
    {
        spawnBlackHole();
    }
    gSpawnBlackHolePressed = bDown;

    bool spaceDown = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
    if (spaceDown && !gPausePressed)
    {
        gPaused = !gPaused;
    }
    gPausePressed = spaceDown;

    const float panStep = 0.30f * (gCameraDistance / 45.0f);
    const glm::vec3 forward = glm::normalize(gCameraTarget - gCameraPos);
    const glm::vec3 right = glm::normalize(glm::cross(forward, gCameraUp));
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
    {
        gCameraTarget += forward * panStep;
    }
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
    {
        gCameraTarget -= forward * panStep;
    }
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
    {
        gCameraTarget -= right * panStep;
    }
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
    {
        gCameraTarget += right * panStep;
    }
    if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
    {
        gCameraTarget += gCameraUp * panStep;
    }
    if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
    {
        gCameraTarget -= gCameraUp * panStep;
    }
    if (glfwGetKey(window, GLFW_KEY_R) == GLFW_PRESS)
    {
        resetCamera();
    }
    if (glfwGetKey(window, GLFW_KEY_EQUAL) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_ADD) == GLFW_PRESS)
    {
        gCameraDistance -= 0.8f;
    }
    if (glfwGetKey(window, GLFW_KEY_MINUS) == GLFW_PRESS || glfwGetKey(window, GLFW_KEY_KP_SUBTRACT) == GLFW_PRESS)
    {
        gCameraDistance += 0.8f;
    }
    updateCameraView();
}

void cursorPosCallback(GLFWwindow*, double xpos, double ypos)
{
    double dx = xpos - gLastMouseX;
    double dy = ypos - gLastMouseY;
    gLastMouseX = xpos;
    gLastMouseY = ypos;

    if (gLeftMousePressed)
    {
        gCameraYaw += static_cast<float>(dx) * 0.18f;
        gCameraPitch -= static_cast<float>(dy) * 0.18f;
        updateCameraView();
    }
    if (gRightMousePressed)
    {
        const float pan = 0.015f * gCameraDistance;
        const glm::vec3 forward = glm::normalize(gCameraTarget - gCameraPos);
        const glm::vec3 right = glm::normalize(glm::cross(forward, gCameraUp));
        gCameraTarget -= right * static_cast<float>(dx) * pan;
        gCameraTarget += gCameraUp * static_cast<float>(dy) * pan;
        updateCameraView();
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        gLeftMousePressed = (action == GLFW_PRESS);
    }
    if (button == GLFW_MOUSE_BUTTON_RIGHT)
    {
        gRightMousePressed = (action == GLFW_PRESS);
    }
    glfwGetCursorPos(window, &gLastMouseX, &gLastMouseY);
}

void scrollCallback(GLFWwindow*, double, double yoffset)
{
    gCameraDistance -= static_cast<float>(yoffset) * 2.2f;
    gFov -= static_cast<float>(yoffset) * 0.6f;
    gFov = std::clamp(gFov, 20.0f, 70.0f);
    updateCameraView();
}

} // namespace

int main()
{
    if (!glfwInit())
    {
        std::cerr << "Failed to initialize GLFW\n";
        return EXIT_FAILURE;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

    GLFWwindow* window = glfwCreateWindow(kWindowWidth, kWindowHeight, "Gravity Simulation - Newton + Relativity-inspired", nullptr, nullptr);
    if (!window)
    {
        std::cerr << "Failed to create window\n";
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetCursorPosCallback(window, cursorPosCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetScrollCallback(window, scrollCallback);

    GLenum glewErr = glewInit();
    if (glewErr != GLEW_OK)
    {
        std::cerr << "Failed to initialize GLEW: " << glewGetErrorString(glewErr) << "\n";
        glfwDestroyWindow(window);
        glfwTerminate();
        return EXIT_FAILURE;
    }

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    gProgram = makeProgram();
    initializeTextures();
    buildSphereMesh(32, 32);
    buildRingMesh(1.15f, 2.05f, 160);
    seedSolarSystem();
    buildGrid();

    gProjection = glm::perspective(glm::radians(gFov), kAspect, 0.1f, 300.0f);
    resetCamera();

    glUseProgram(gProgram);
    GLint modelLoc = glGetUniformLocation(gProgram, "uModel");
    GLint viewLoc = glGetUniformLocation(gProgram, "uView");
    GLint projectionLoc = glGetUniformLocation(gProgram, "uProjection");
    GLint colorLoc = glGetUniformLocation(gProgram, "uColor");
    GLint lightLoc = glGetUniformLocation(gProgram, "uLightPos");
    GLint viewPosLoc = glGetUniformLocation(gProgram, "uViewPos");
    GLint emissiveLoc = glGetUniformLocation(gProgram, "uEmissive");
    GLint useTextureLoc = glGetUniformLocation(gProgram, "uUseTexture");
    GLint alphaLoc = glGetUniformLocation(gProgram, "uAlpha");
    GLint texLoc = glGetUniformLocation(gProgram, "uTex");
    glUniform1i(texLoc, 0);

    std::cout << "Controls:\n";
    std::cout << "  P: Spawn a random planet\n";
    std::cout << "  B: Spawn a black hole\n";
    std::cout << "  SPACE: Pause/Resume simulation\n";
    std::cout << "  Mouse Left Drag: Orbit camera\n";
    std::cout << "  Mouse Right Drag: Pan camera\n";
    std::cout << "  Mouse Wheel / +/-: Zoom\n";
    std::cout << "  WASDQE: Pan/Fly camera target\n";
    std::cout << "  R: Reset camera\n";
    std::cout << "  ESC: Exit\n";

    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        if (!gPaused)
        {
            integratePhysics();
            buildGrid();
        }

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        int fbWidth = 0;
        int fbHeight = 0;
        glfwGetFramebufferSize(window, &fbWidth, &fbHeight);
        glViewport(0, 0, fbWidth, fbHeight);
        float aspect = fbHeight > 0 ? static_cast<float>(fbWidth) / static_cast<float>(fbHeight) : kAspect;
        gProjection = glm::perspective(glm::radians(gFov), aspect, 0.1f, 300.0f);

        glUseProgram(gProgram);
        glUniformMatrix4fv(viewLoc, 1, GL_FALSE, glm::value_ptr(gView));
        glUniformMatrix4fv(projectionLoc, 1, GL_FALSE, glm::value_ptr(gProjection));
        glUniform3fv(lightLoc, 1, glm::value_ptr(gLightPos));
        glUniform3fv(viewPosLoc, 1, glm::value_ptr(gCameraPos));

        drawGrid(modelLoc, colorLoc, emissiveLoc, useTextureLoc, alphaLoc);
        drawTrails(modelLoc, colorLoc, emissiveLoc, useTextureLoc, alphaLoc);

        for (const auto& b : gBodies)
        {
            drawBody(b, modelLoc, colorLoc, emissiveLoc, useTextureLoc, alphaLoc);
            if (b.name == "SATURN")
            {
                drawSaturnRing(b, modelLoc, colorLoc, emissiveLoc, useTextureLoc, alphaLoc);
            }
        }
        drawBodyLabels(window);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    if (gGridVbo != 0)
    {
        glDeleteBuffers(1, &gGridVbo);
    }
    if (gGridVao != 0)
    {
        glDeleteVertexArrays(1, &gGridVao);
    }
    if (gRingEbo != 0)
    {
        glDeleteBuffers(1, &gRingEbo);
    }
    if (gRingVbo != 0)
    {
        glDeleteBuffers(1, &gRingVbo);
    }
    if (gRingVao != 0)
    {
        glDeleteVertexArrays(1, &gRingVao);
    }
    if (gSphereEbo != 0)
    {
        glDeleteBuffers(1, &gSphereEbo);
    }
    if (gSphereVbo != 0)
    {
        glDeleteBuffers(1, &gSphereVbo);
    }
    if (gSphereVao != 0)
    {
        glDeleteVertexArrays(1, &gSphereVao);
    }
    if (gProgram != 0)
    {
        glDeleteProgram(gProgram);
    }
    for (auto& [_, tex] : gTextureCache)
    {
        if (tex != 0)
        {
            glDeleteTextures(1, &tex);
        }
    }
    if (gFallbackTexture != 0)
    {
        glDeleteTextures(1, &gFallbackTexture);
    }
    for (GLuint tex : gDynamicTextures)
    {
        if (tex != 0)
        {
            glDeleteTextures(1, &tex);
        }
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return EXIT_SUCCESS;
}
