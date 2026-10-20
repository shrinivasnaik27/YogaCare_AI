"""
YogaCare AI - Dynamic Pose Reference Configuration
Provides structured biomechanical references, instructions, alignment tips,
difficulty ratings, and verified image assets for all supported yoga asanas.
"""

from dataclasses import dataclass, field
import os
from typing import Dict, List, Optional, Tuple


@dataclass
class PoseReference:
    """Structured reference information for a yoga asana."""
    pose_id: str
    name: str
    sanskrit_name: str
    difficulty: str  # "Beginner", "Intermediate", "Advanced"
    description: str
    local_image_path: str
    fallback_image_url: str
    instructions: List[str] = field(default_factory=list)
    alignment_tips: List[str] = field(default_factory=list)
    target_joints: Dict[str, str] = field(default_factory=dict)


# Registry of all supported yoga posture references
POSE_REFERENCES: Dict[str, PoseReference] = {
    "warrior_ii": PoseReference(
        pose_id="warrior_ii",
        name="Warrior II",
        sanskrit_name="Virabhadrasana II",
        difficulty="Beginner to Intermediate",
        description="A foundational standing posture that builds lower body power, opens the hips and chest, and develops laser-focused concentration.",
        local_image_path=os.path.join("assets", "reference_poses", "warrior_ii.jpg"),
        fallback_image_url="https://upload.wikimedia.org/wikipedia/commons/3/31/Ashish_veerbhadrasana.jpg",
        instructions=[
            "Stand in a wide lateral stance (roughly 3 to 4 feet apart).",
            "Turn your front foot out 90 degrees; angle your back foot slightly inward (about 15 degrees).",
            "Bend your front knee directly over your front ankle until your front thigh is nearly parallel to the mat (aim for 90°).",
            "Keep your back leg completely straight, pressing firmly into the outer edge of your back heel.",
            "Extend both arms horizontally at shoulder height, reaching actively forward and backward.",
            "Stack your torso vertically directly over your pelvis; gaze gently over your front middle finger.",
        ],
        alignment_tips=[
            "💡 Knee Safety: Stack front knee directly above the ankle; never let it collapse inward.",
            "🎯 Arm Line: Keep both arms parallel to the floor without shrugging shoulders.",
            "⚖️ Torso Neutrality: Resist leaning forward over the front thigh; keep spine vertical.",
            "🦶 Grounding: Press firmly through the back foot arch to engage the posterior chain.",
        ],
        target_joints={
            "Front Knee": "90° (Range: 80°–105°)",
            "Back Knee": "175° (Straight pillar: 160°–180°)",
            "Arms": "Horizontal (<= 15° tilt from parallel)",
            "Torso": "Vertical (80°–100° with hips)",
        },
    ),
    "tree_pose": PoseReference(
        pose_id="tree_pose",
        name="Tree Pose",
        sanskrit_name="Vrikshasana",
        difficulty="Beginner",
        description="A classic standing balance asana that develops unilateral leg stability, hip mobility, neuro-muscular coordination, and mental equilibrium.",
        local_image_path=os.path.join("assets", "reference_poses", "tree_pose.jpg"),
        fallback_image_url="https://upload.wikimedia.org/wikipedia/commons/1/1c/Flexible_Tree_yoga_series_by_Dr_Igor_Popov.jpg",
        instructions=[
            "Begin standing tall in Mountain Pose with weight evenly distributed on both feet.",
            "Shift your weight onto your standing leg, pressing all four corners of the foot firmly into the ground.",
            "Bend your opposite knee and place the sole of the foot on your inner thigh or inner calf.",
            "Rotate the bent knee outward to open the hip, keeping your pelvis squared forward.",
            "Bring your palms together at your heart in prayer (Anjali Mudra) or extend arms straight overhead.",
            "Fix your gaze (Drishti) on an unmoving point at eye level to stabilize balance.",
        ],
        alignment_tips=[
            "⚠️ CRITICAL SAFETY: Never place your foot directly on the knee joint capsule! Place on inner thigh or calf.",
            "🌲 Standing Leg: Keep the standing knee straight (~175°) without hyperextending backward.",
            "🧘 Pelvic Level: Keep both hip bones level horizontally; avoid hiking the lifted hip.",
            "💨 Breath: Maintain slow, diaphragmatic breathing to stabilize neuromuscular wobble.",
        ],
        target_joints={
            "Standing Knee": "175° (Straight pillar: 165°–180°)",
            "Lifted Knee": "Flexed & rotated outward (25°–65°)",
            "Spine": "Vertical (<= 12° deviation)",
            "Foot Safety": "Above or below knee joint",
        },
    ),
    "downward_dog": PoseReference(
        pose_id="downward_dog",
        name="Downward-Facing Dog",
        sanskrit_name="Adho Mukha Svanasana",
        difficulty="Beginner to Intermediate",
        description="An energizing inversion and full-body stretch that decompresses the spine, opens hamstrings and calves, and strengthens the shoulders.",
        local_image_path=os.path.join("assets", "reference_poses", "downward_dog.jpg"),
        fallback_image_url="https://upload.wikimedia.org/wikipedia/commons/5/57/Downward-Facing-Dog.JPG",
        instructions=[
            "Start on your hands and knees on the mat with wrists beneath shoulders and knees beneath hips.",
            "Spread your fingers wide, press firmly through knuckles and index fingers, and tuck your toes.",
            "Exhale and lift your knees off the mat, pressing your sitting bones up and back toward the ceiling.",
            "Form a clean inverted 'V' shape with your body.",
            "Press your chest gently toward your thighs to lengthen your spine.",
            "Draw your heels toward the mat, keeping knees straight or with a micro-bend if hamstrings are tight.",
        ],
        alignment_tips=[
            "🐕 Inverted V: Hips form the highest apex point (ideal hip angle: 65°–88°).",
            "👐 Hand Press: Distribute weight evenly across palms to avoid wrist compression.",
            "🦴 Straight Spine: Prioritize a straight, long spine over forcing heels to touch the mat.",
            "👂 Neck Alignment: Relax the neck; let ears rest parallel between upper arms.",
        ],
        target_joints={
            "Hip Apex": "65°–88° (Inverted V)",
            "Elbows": "165°–180° (Straight arms)",
            "Knees": "155°–180° (Extended legs)",
        },
    ),
    "plank_pose": PoseReference(
        pose_id="plank_pose",
        name="Plank Pose",
        sanskrit_name="Phalakasana",
        difficulty="Intermediate",
        description="A foundational core-strengthening posture that engages the transversus abdominis, shoulders, chest, and posterior kinetic chain.",
        local_image_path=os.path.join("assets", "reference_poses", "plank_pose.jpg"),
        fallback_image_url="https://upload.wikimedia.org/wikipedia/commons/6/61/Chaturanga-Dandasana_high_Yoga-Asana_Nina-Mel.jpg",
        instructions=[
            "From tabletop or Downward Dog, step both feet back until legs are fully extended.",
            "Stack wrists directly under shoulders with fingers spread wide.",
            "Draw your navel firmly toward your spine to engage your deep core.",
            "Create a single, unbroken straight diagonal line from the crown of your head to your heels.",
            "Press actively through your heels and firm your quadriceps and glutes.",
            "Keep your neck neutral by gazing at the mat about a foot ahead of your hands.",
        ],
        alignment_tips=[
            "⚠️ Avoid Lumbar Sag: Do not let your hips sink toward the mat; this strains the lower back.",
            "🪵 Neutral Plane: Maintain a 165°–180° straight line from shoulders through hips to ankles.",
            "💪 Arm Stack: Lock wrists directly under shoulder joints; push the floor away.",
            "🛡️ Shoulder Protraction: Broaden across the upper back to support the thoracic spine.",
        ],
        target_joints={
            "Shoulder-Hip-Ankle": "165°–180° (Straight body line)",
            "Elbows": "155°–180° (Extended arms)",
            "Core Elevation": "Hips elevated above floor plane",
        },
    ),
    "cobra_pose": PoseReference(
        pose_id="cobra_pose",
        name="Cobra Pose",
        sanskrit_name="Bhujangasana",
        difficulty="Beginner to Intermediate",
        description="A gentle prone backbend that strengthens the erector spinae, opens the anterior chest and shoulders, and enhances spinal flexibility.",
        local_image_path=os.path.join("assets", "reference_poses", "cobra_pose.jpg"),
        fallback_image_url="https://upload.wikimedia.org/wikipedia/commons/a/ae/Bhujangasana_Yoga-Asana_Nina-Mel.jpg",
        instructions=[
            "Lie flat on your stomach with tops of feet pressed firmly into the mat, hip-width apart.",
            "Place palms flat on the mat directly under your shoulders, hugging elbows close to your torso.",
            "Inhale and slowly lift your head, chest, and upper abdomen off the mat using back muscles.",
            "Keep a soft micro-bend in your elbows; do not lock arms or push all your weight into hands.",
            "Keep your pelvis, hips, and thighs grounded on the floor.",
            "Roll your shoulders down and back away from your ears, broadening across your collarbones.",
        ],
        alignment_tips=[
            "🐍 Elbow Micro-Bend: Keep elbows bent at 120°–160° hugging the ribcage; avoid locking out arms.",
            "🦴 Pelvic Grounding: Keep pubic bone and hips anchored to prevent compressing lumbar vertebrae.",
            "🫀 Thoracic Extension: Focus the backbend in the upper and middle back rather than the lower back.",
            "👀 Neck Safety: Gaze gently forward or slightly upward without cranking the cervical spine.",
        ],
        target_joints={
            "Thoracic Lift": "125°–150° (Chest arch)",
            "Elbow Angle": "120°–160° (Micro-bend)",
            "Pelvis": "Grounded to mat",
        },
    ),
    "mountain_pose": PoseReference(
        pose_id="mountain_pose",
        name="Mountain Pose",
        sanskrit_name="Tadasana",
        difficulty="Beginner",
        description="The foundational blueprint for all standing asanas, teaching structural alignment, symmetrical posture, grounded poise, and mindful presence.",
        local_image_path=os.path.join("assets", "reference_poses", "mountain_pose.jpg"),
        fallback_image_url="https://upload.wikimedia.org/wikipedia/commons/6/6f/Tadasana_Yoga-Asana_Nina-Mel.jpg",
        instructions=[
            "Stand upright with big toes touching and heels slightly separated, or feet hip-distance apart.",
            "Lift and spread your toes, then place them down evenly, distributing weight across all four corners of each foot.",
            "Engage your quadriceps and lift your kneecaps without locking the knees.",
            "Tuck your tailbone slightly and draw your lower belly gently in and up.",
            "Roll your shoulders up, back, and down, letting your arms rest naturally at your sides with palms facing forward.",
            "Lengthen through the crown of your head, keeping your chin parallel to the floor.",
        ],
        alignment_tips=[
            "⛰️ Four-Point Balance: Feel equal pressure on the ball of the big toe, little toe, inner heel, and outer heel.",
            "📏 Vertical Plumb Line: Ears, shoulders, hips, and ankles should align along a single vertical axis.",
            "🧘 Neutral Pelvis: Avoid excessive anterior pelvic tilt (arching lower back) or rounding.",
            "🌬️ Relaxed Jaw & Neck: Release tension from facial muscles and breathe naturally.",
        ],
        target_joints={
            "Knees": "165°–180° (Straight & active)",
            "Hips": "165°–180° (Neutral extension)",
            "Spine": "Vertical (<= 12° deviation)",
        },
    ),
}


def get_pose_reference(pose_id: str) -> PoseReference:
    """
    Retrieves the structured PoseReference for a given pose_id.
    Falls back to Mountain Pose if the key is unknown.
    """
    key = pose_id.lower().strip()
    if key in POSE_REFERENCES:
        return POSE_REFERENCES[key]
    # Return first available or mountain pose
    return POSE_REFERENCES.get("mountain_pose", list(POSE_REFERENCES.values())[0])


def get_all_pose_references() -> Dict[str, PoseReference]:
    """Returns the full dictionary of supported pose references."""
    return POSE_REFERENCES


def resolve_reference_image(pose_ref: PoseReference) -> Tuple[Optional[str], bool, str]:
    """
    Resolves the best available image source for a pose reference.

    Returns:
        (image_source, is_local_file, status_message)
        - image_source: local file path, or URL, or None if unavailable.
        - is_local_file: True if image_source is a verified local file on disk.
        - status_message: explanation (e.g. "Loaded from local assets", "Using fallback URL", etc.).
    """
    # 1. Try local image file
    if os.path.exists(pose_ref.local_image_path) and os.path.getsize(pose_ref.local_image_path) > 1024:
        return pose_ref.local_image_path, True, f"Verified local reference: {pose_ref.local_image_path}"

    # 2. Try fallback URL
    if pose_ref.fallback_image_url and pose_ref.fallback_image_url.startswith("http"):
        return pose_ref.fallback_image_url, False, f"Using verified remote reference URL for {pose_ref.name}"

    # 3. Missing both
    return (
        None,
        False,
        f"Reference image missing for {pose_ref.name}. Please place '{os.path.basename(pose_ref.local_image_path)}' in 'assets/reference_poses/'."
    )
