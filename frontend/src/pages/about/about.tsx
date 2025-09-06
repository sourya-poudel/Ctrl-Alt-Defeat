import { useNavigate } from 'react-router-dom';
import React, { useState } from "react";
import { motion } from "framer-motion";
import {

ShieldCheck,
Camera,
Image,
FileText,
Users,
Github,
Linkedin,
Mail,
Phone,
MapPin,
Globe,
Sun,
Moon,
Languages,
Star,
ChevronDown,
ChevronUp,
Newspaper,
Download,
Lock,
Info,
HeartHandshake,
ArrowRight,
ArrowLeft,
ArrowUpRight,
} from "lucide-react";

// Dummy data for demonstration
const techStack = [
{ name: "React", color: "bg-blue-500", icon: <Globe size={18} /> },
{ name: "FastAPI", color: "bg-green-500", icon: <ShieldCheck size={18} /> },
{ name: "PyTorch", color: "bg-red-500", icon: <Image size={18} /> },
{ name: "Firebase", color: "bg-yellow-400", icon: <FileText size={18} /> },
];




const teamMembers = [
{
    name: "Alice Johnson",
    role: "Lead Developer",
    bio: "Expert in AI and full-stack development.",
    photo: "https://randomuser.me/api/portraits/women/44.jpg",
    linkedin: "#",
    github: "#",
},
{
    name: "Bob Smith",
    role: "Backend Engineer",
    bio: "Specializes in scalable APIs and data pipelines.",
    photo: "https://randomuser.me/api/portraits/men/32.jpg",
    linkedin: "#",
    github: "#",
},
{
    name: "Carol Lee",
    role: "UI/UX Designer",
    bio: "Passionate about beautiful, accessible interfaces.",
    photo: "https://randomuser.me/api/portraits/women/68.jpg",
    linkedin: "#",
    github: "#",
},
];

const goals = [
"Empower communities with AI-driven safety tools.",
"Ensure privacy and ethical data handling.",
"Deliver real-time insights from CCTV and records.",
"Foster collaboration with law enforcement and partners.",
];

const milestones = [
{ date: "Q1 2024", event: "Platform MVP Launch" },
{ date: "Q2 2024", event: "AI Sketch-to-Image Beta" },
{ date: "Q3 2024", event: "Partner Integrations" },
{ date: "Q4 2024", event: "Mobile App Release" },
];

const faqs = [
{
    question: "How does AI criminal detection work?",
    answer:
        "Our platform uses advanced machine learning models to analyze CCTV footage and identify suspicious activities in real-time.",
},
{
    question: "Is my data safe?",
    answer:
        "We use end-to-end encryption and comply with GDPR to ensure your data is secure and private.",
},
{
    question: "Can I integrate with other systems?",
    answer:
        "Yes, our API allows seamless integration with third-party tools and platforms.",
},
];

const partners = [
{
    name: "SafeCity",
    logo: "https://dummyimage.com/80x40/4f46e5/fff.png&text=SafeCity",
    desc: "Urban safety analytics partner.",
},
{
    name: "OpenAI",
    logo: "https://dummyimage.com/80x40/10b981/fff.png&text=OpenAI",
    desc: "AI research and consulting.",
},
];

const press = [
{
    title: "Platform Revolutionizes CCTV Analysis",
    link: "#",
    date: "2024-05-10",
    outlet: "TechNews",
},
{
    title: "AI for Safer Cities: Interview with Founders",
    link: "#",
    date: "2024-06-01",
    outlet: "City Journal",
},
];

const testimonials = [
{
    name: "Detective Mark",
    review:
        "The platform helped us solve cases faster and improved our workflow.",
    rating: 5,
    photo: "https://randomuser.me/api/portraits/men/45.jpg",
},
{
    name: "Officer Jane",
    review:
        "AI-powered CCTV analysis is a game changer for public safety.",
    rating: 4,
    photo: "https://randomuser.me/api/portraits/women/23.jpg",
},
];

const stats = [
{ label: "Users", value: 1200 },
{ label: "Cases Solved", value: 340 },
{ label: "Partners", value: 12 },
];

const languages = ["English", "Español", "Français", "中文"];

function AnimatedCounter({ value }: { value: number }) {
const [display, setDisplay] = useState(0);
React.useEffect(() => {
    let start = 0;
    const end = value;
    if (start === end) return;
    let duration = 1000;
    let increment = end / (duration / 16);
    let current = start;
    const timer = setInterval(() => {
        current += increment;
        if (current >= end) {
            current = end;
            clearInterval(timer);
        }
        setDisplay(Math.floor(current));
    }, 16);
    return () => clearInterval(timer);
}, [value]);
return (
    <motion.span
        initial={{ scale: 0.8 }}
        animate={{ scale: 1 }}
        transition={{ duration: 0.5 }}
        className="font-bold text-3xl text-indigo-600"
    >
        {display}
    </motion.span>
);
}

function Accordion({ items }: { items: typeof faqs }) {
const [openIdx, setOpenIdx] = useState<number | null>(null);
return (
    <div className="space-y-2">
        {items.map((item, idx) => (
            <div key={idx} className="bg-white/80 rounded-lg shadow">
                <button
                    className="flex w-full justify-between items-center px-4 py-3 text-lg font-semibold"
                    onClick={() => setOpenIdx(openIdx === idx ? null : idx)}
                    aria-expanded={openIdx === idx}
                >
                    <span>{item.question}</span>
                    {openIdx === idx ? <ChevronUp /> : <ChevronDown />}
                </button>
                <motion.div
                    initial={false}
                    animate={{ height: openIdx === idx ? "auto" : 0, opacity: openIdx === idx ? 1 : 0 }}
                    transition={{ duration: 0.3 }}
                    className="overflow-hidden px-4 pb-3 text-gray-700"
                >
                    {openIdx === idx && <div>{item.answer}</div>}
                </motion.div>
            </div>
        ))}
    </div>
);
}

function About() {
const navigate = useNavigate();    
const [theme, setTheme] = useState<"dark" | "light">("dark");
const [lang, setLang] = useState(languages[0]);
const [form, setForm] = useState({ name: "", email: "", message: "" });
const [formError, setFormError] = useState("");
const [formSuccess, setFormSuccess] = useState(false);

const handleThemeSwitch = () => setTheme(theme === "dark" ? "light" : "dark");
const handleLangChange = (e: React.ChangeEvent<HTMLSelectElement>) => setLang(e.target.value);

const handleFormChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setFormError("");
    setFormSuccess(false);
};

const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name || !form.email || !form.message) {
        setFormError("Please fill in all fields.");
        return;
    }
    setFormSuccess(true);
    setForm({ name: "", email: "", message: "" });
};

return (
    <div
        className={`min-h-screen w-full font-sans transition-colors duration-500 ${
            theme === "dark"
                ? "bg-gradient-to-br from-gray-900 via-indigo-900 to-gray-800 text-gray-100"
                : "bg-gradient-to-br from-indigo-100 via-white to-purple-100 text-gray-900"
        }`}
    >
        {/* Responsive Navbar */}
        <nav className="flex items-center justify-between px-6 py-4 bg-white/60 dark:bg-gray-900/60 rounded-b-2xl shadow-lg sticky top-0 z-20">
            <div className="flex items-center gap-2">
                            <motion.button
                                whileHover={{ scale: 1.05 }}
                                whileTap={{ scale: 0.95 }}
                                onClick={() => navigate('/home')}
                                className="flex items-center gap-2 px-4 py-2 bg-blue-900/80 rounded-xl hover:bg-blue-800 transition-colors shadow"
                            >
                                <ArrowLeft className="w-5 h-5" />
                                <span>Home</span>
                            </motion.button>
                
            </div>
            <div className="flex items-center gap-4">
                <button
                    aria-label="Switch theme"
                    onClick={handleThemeSwitch}
                    className="p-2 rounded-full hover:bg-indigo-100 dark:hover:bg-indigo-900 transition"
                >
                    {theme === "dark" ? <Sun /> : <Moon />}
                </button>
                <select
                    aria-label="Select language"
                    value={lang}
                    onChange={handleLangChange}
                    className="bg-transparent border-none text-lg"
                >
                    {languages.map((l) => (
                        <option key={l} value={l}>
                            {l}
                        </option>
                    ))}
                </select>
            </div>
        </nav>

        {/* Animated Heading */}
        <motion.header
            initial={{ opacity: 0, y: -40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-4xl mx-auto mt-10 mb-8 text-center"
        >
            <h1 className="text-5xl font-extrabold bg-gradient-to-r from-indigo-600 via-purple-500 to-pink-500 bg-clip-text text-transparent mb-4 animate-pulse">
                About Us
            </h1>
            <p className="text-xl text-gray-700 dark:text-gray-300">
                Empowering communities with AI-driven safety, privacy, and innovation.
            </p>
        </motion.header>

        {/* Animated Statistics */}
        <motion.section
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="flex flex-wrap justify-center gap-8 mb-12"
        >
            {stats.map((stat) => (
                <div
                    key={stat.label}
                    className="flex flex-col items-center bg-white/80 dark:bg-gray-800/80 rounded-xl shadow-lg p-6 w-40"
                >
                    <AnimatedCounter value={stat.value} />
                    <span className="mt-2 text-lg font-medium">{stat.label}</span>
                </div>
            ))}
        </motion.section>

        {/* Platform Overview */}
        <motion.section
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto bg-white/90 dark:bg-gray-900/90 rounded-3xl shadow-xl p-8 mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <ShieldCheck className="text-indigo-600" /> Platform Overview
            </h2>
            <p className="mb-4 text-lg">
                Our mission is to leverage artificial intelligence for safer, smarter cities. We provide real-time criminal detection, advanced CCTV analysis, sketch-to-image conversion, and robust record management.
            </p>
            <div className="flex flex-wrap gap-6 mb-6">
                <div className="flex items-center gap-2">
                    <Camera className="text-purple-500" /> CCTV Analysis
                </div>
                <div className="flex items-center gap-2">
                    <Image className="text-pink-500" /> Sketch-to-Image
                </div>
                <div className="flex items-center gap-2">
                    <FileText className="text-indigo-500" /> Record Management
                </div>
                <div className="flex items-center gap-2">
                    <ShieldCheck className="text-green-500" /> AI Criminal Detection
                </div>
            </div>
            <div className="flex flex-wrap gap-3">
                {techStack.map((tech) => (
                    <span
                        key={tech.name}
                        className={`inline-flex items-center gap-1 px-3 py-1 rounded-full text-sm font-semibold ${tech.color} text-white shadow`}
                        title={tech.name}
                    >
                        {tech.icon} {tech.name}
                    </span>
                ))}
            </div>
        </motion.section>

        {/* Team Members */}
        <motion.section
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto mb-12"
        >
            <h2 className="text-3xl font-bold mb-6 flex items-center gap-2">
                <Users className="text-indigo-600" /> Meet the Team
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                {teamMembers.map((member) => (
                    <motion.div
                        key={member.name}
                        whileHover={{ scale: 1.05, boxShadow: "0 8px 32px rgba(99,102,241,0.15)" }}
                        className="bg-white/90 dark:bg-gray-900/90 rounded-2xl shadow-lg p-6 flex flex-col items-center"
                    >
                        <img
                            src={member.photo}
                            alt={member.name}
                            className="w-24 h-24 rounded-full mb-4 object-cover border-4 border-indigo-400"
                        />
                        <h3 className="text-xl font-bold">{member.name}</h3>
                        <span className="text-indigo-600 font-medium mb-2">{member.role}</span>
                        <p className="text-center text-gray-700 dark:text-gray-300 mb-3">{member.bio}</p>
                        <div className="flex gap-3">
                            <a href={member.linkedin} target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                                <Linkedin className="hover:text-blue-600 transition" />
                            </a>
                            <a href={member.github} target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                                <Github className="hover:text-gray-800 dark:hover:text-gray-200 transition" />
                            </a>
                        </div>
                    </motion.div>
                ))}
            </div>
        </motion.section>

        {/* Project Goals & Values */}
        <motion.section
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto bg-white/90 dark:bg-gray-900/90 rounded-3xl shadow-xl p-8 mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <HeartHandshake className="text-pink-500" /> Project Goals & Values
            </h2>
            <ul className="list-disc ml-6 mb-6 text-lg">
                {goals.map((goal) => (
                    <li key={goal}>{goal}</li>
                ))}
            </ul>
            <div className="mb-4">
                <h3 className="font-semibold text-lg mb-2">Roadmap</h3>
                <div className="flex flex-col gap-2">
                    {milestones.map((m) => (
                        <div key={m.date} className="flex items-center gap-2">
                            <Star className="text-yellow-400" size={18} />
                            <span className="font-medium">{m.date}:</span>
                            <span>{m.event}</span>
                        </div>
                    ))}
                </div>
            </div>
        </motion.section>

        {/* Contact Information & Form */}
        <motion.section
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto grid md:grid-cols-2 gap-8 mb-12"
        >
            <div>
                <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                    <Mail className="text-indigo-600" /> Contact Us
                </h2>
                <div className="mb-4 space-y-2">
                    <div className="flex items-center gap-2">
                        <Mail /> <span>info@aisafety.com</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Phone /> <span>+1 234 567 8901</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <MapPin /> <span>123 Innovation Ave, Tech City</span>
                    </div>
                </div>
                {/* Optional Map Embed */}
                <iframe
                    title="Location Map"
                    src="https://www.openstreetmap.org/export/embed.html?bbox=0,0,0,0"
                    className="w-full h-32 rounded-lg border"
                    loading="lazy"
                />
            </div>
            <form
                className="bg-white/80 dark:bg-gray-800/80 rounded-2xl shadow-lg p-6 flex flex-col gap-4"
                onSubmit={handleFormSubmit}
            >
                <h3 className="font-semibold text-lg mb-2">Send us a message</h3>
                <input
                    name="name"
                    type="text"
                    placeholder="Your Name"
                    value={form.name}
                    onChange={handleFormChange}
                    className="p-2 rounded border focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    required
                />
                <input
                    name="email"
                    type="email"
                    placeholder="Your Email"
                    value={form.email}
                    onChange={handleFormChange}
                    className="p-2 rounded border focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    required
                />
                <textarea
                    name="message"
                    placeholder="Your Message"
                    value={form.message}
                    onChange={handleFormChange}
                    className="p-2 rounded border focus:outline-none focus:ring-2 focus:ring-indigo-400"
                    rows={3}
                    required
                />
                {formError && <div className="text-red-500">{formError}</div>}
                {formSuccess && (
                    <motion.div
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="text-green-600"
                    >
                        Message sent! Thank you.
                    </motion.div>
                )}
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    type="submit"
                    className="bg-indigo-600 text-white font-bold py-2 rounded-lg shadow hover:bg-indigo-700 transition"
                >
                    Send <ArrowUpRight className="inline ml-1" />
                </motion.button>
            </form>
        </motion.section>

        {/* FAQs & Help */}
        <motion.section
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto bg-white/90 dark:bg-gray-900/90 rounded-3xl shadow-xl p-8 mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <Info className="text-indigo-600" /> FAQs & Help
            </h2>
            <Accordion items={faqs} />
            <div className="mt-6 flex gap-4">
                <a
                    href="#"
                    className="text-indigo-600 hover:underline flex items-center gap-1"
                >
                    Documentation <ArrowRight size={16} />
                </a>
                <a
                    href="#"
                    className="text-indigo-600 hover:underline flex items-center gap-1"
                >
                    Support <ArrowRight size={16} />
                </a>
            </div>
        </motion.section>

        {/* Partners & Collaborators */}
        <motion.section
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <Users className="text-green-500" /> Partners & Collaborators
            </h2>
            <div className="flex flex-wrap gap-8">
                {partners.map((p) => (
                    <div
                        key={p.name}
                        className="flex items-center gap-4 bg-white/80 dark:bg-gray-800/80 rounded-xl shadow p-4"
                    >
                        <img
                            src={p.logo}
                            alt={p.name}
                            className="w-20 h-10 object-contain rounded"
                        />
                        <div>
                            <div className="font-bold">{p.name}</div>
                            <div className="text-gray-600 dark:text-gray-300 text-sm">{p.desc}</div>
                        </div>
                    </div>
                ))}
            </div>
        </motion.section>

        {/* Press & Media */}
        <motion.section
            initial={{ opacity: 0, x: 40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <Newspaper className="text-indigo-600" /> Press & Media
            </h2>
            <div className="flex flex-col gap-4">
                {press.map((article) => (
                    <a
                        key={article.title}
                        href={article.link}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-3 bg-white/80 dark:bg-gray-800/80 rounded-lg shadow p-4 hover:bg-indigo-50 dark:hover:bg-indigo-900 transition"
                    >
                        <Newspaper className="text-indigo-500" />
                        <div>
                            <div className="font-semibold">{article.title}</div>
                            <div className="text-sm text-gray-500">
                                {article.outlet} &middot; {article.date}
                            </div>
                        </div>
                    </a>
                ))}
            </div>
            <a
                href="#"
                className="mt-4 inline-flex items-center gap-2 text-indigo-600 hover:underline"
            >
                <Download /> Download Media Kit
            </a>
        </motion.section>

        {/* Legal & Privacy */}
        <motion.section
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto bg-white/90 dark:bg-gray-900/90 rounded-3xl shadow-xl p-8 mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <Lock className="text-indigo-600" /> Legal & Privacy
            </h2>
            <div className="mb-2">
                <a href="#" className="text-indigo-600 hover:underline mr-4">
                    Privacy Policy
                </a>
                <a href="#" className="text-indigo-600 hover:underline mr-4">
                    Terms of Service
                </a>
                <a href="#" className="text-indigo-600 hover:underline">
                    Data Protection
                </a>
            </div>
            <p className="text-gray-700 dark:text-gray-300 mt-2">
                We prioritize user privacy and data protection. All data is encrypted and handled in compliance with international standards.
            </p>
        </motion.section>

        {/* User Testimonials */}
        <motion.section
            initial={{ opacity: 0, x: -40 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto mb-12"
        >
            <h2 className="text-3xl font-bold mb-4 flex items-center gap-2">
                <Star className="text-yellow-400" /> User Testimonials
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                {testimonials.map((t) => (
                    <motion.div
                        key={t.name}
                        whileHover={{ scale: 1.03 }}
                        className="bg-white/90 dark:bg-gray-900/90 rounded-2xl shadow-lg p-6 flex flex-col items-center"
                    >
                        <img
                            src={t.photo}
                            alt={t.name}
                            className="w-16 h-16 rounded-full mb-3 object-cover border-2 border-yellow-400"
                        />
                        <div className="font-bold">{t.name}</div>
                        <div className="flex gap-1 mt-1 mb-2">
                            {[...Array(t.rating)].map((_, i) => (
                                <Star key={i} className="text-yellow-400" size={18} />
                            ))}
                        </div>
                        <p className="text-center text-gray-700 dark:text-gray-300">{t.review}</p>
                    </motion.div>
                ))}
            </div>
        </motion.section>

        {/* Call to Action */}
        <motion.section
            initial={{ opacity: 0, y: 40 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.7 }}
            className="max-w-5xl mx-auto text-center mb-12"
        >
            <h2 className="text-3xl font-bold mb-4">Ready to get started?</h2>
            <div className="flex flex-wrap justify-center gap-6 mb-4">
                <a
                    href="#"
                    className="bg-indigo-600 text-white font-bold py-2 px-6 rounded-lg shadow hover:bg-indigo-700 transition"
                >
                    Sign Up
                </a>
                <a
                    href="#"
                    className="bg-pink-500 text-white font-bold py-2 px-6 rounded-lg shadow hover:bg-pink-600 transition"
                >
                    Join the Team
                </a>
                <a
                    href="#"
                    className="bg-green-500 text-white font-bold py-2 px-6 rounded-lg shadow hover:bg-green-600 transition"
                >
                    Follow Us
                </a>
            </div>
            <form className="max-w-md mx-auto flex gap-2 mt-4">
                <input
                    type="email"
                    placeholder="Subscribe to newsletter"
                    className="flex-1 p-2 rounded-l border focus:outline-none focus:ring-2 focus:ring-indigo-400"
                />
                <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    type="submit"
                    className="bg-indigo-600 text-white font-bold px-4 rounded-r shadow hover:bg-indigo-700 transition"
                >
                    Subscribe
                </motion.button>
            </form>
        </motion.section>

        {/* Footer */}
        <footer className="bg-white/80 dark:bg-gray-900/80 rounded-t-2xl shadow-lg py-6 mt-12">
            <div className="max-w-5xl mx-auto flex flex-col md:flex-row justify-between items-center gap-4 px-4">
                <div className="flex items-center gap-2">
                    <Camera className="text-indigo-600" />
                    <span className="font-bold">AI Safety Platform</span>
                </div>
                <div className="flex gap-4">
                    <a href="#" aria-label="GitHub">
                        <Github className="hover:text-gray-800 dark:hover:text-gray-200 transition" />
                    </a>
                    <a href="#" aria-label="LinkedIn">
                        <Linkedin className="hover:text-blue-600 transition" />
                    </a>
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">
                    &copy; {new Date().getFullYear()} AI Safety Platform. All rights reserved.
                </div>
            </div>
        </footer>
    </div>
);
}

export default About;