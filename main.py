#!/usr/bin/env python3
"""
Tetra detect
"""

import numpy as np
from rtlsdr import RtlSdr
import time
import threading
import pygame
from scipy.signal import welch, find_peaks
from collections import deque
import argparse

class TetraDetector:
    def __init__(self, center_freq=390e6, sample_rate=2.4e6, 
                 threshold_db=12, alarm_sound=None, sensitivity='medium'):
        """
        TETRA detector met alarm functionaliteit
        
        Args:
            center_freq: Center frequentie (390 MHz is midden C2000 band)
            sample_rate: Sample rate (2.4 MS/s is max stabiel voor RTL-SDR)
            threshold_db: dB boven ruisvloer voor detectie (lager = gevoeliger)
            alarm_sound: Pad naar .wav bestand voor alarm (None = beep)
            sensitivity: 'low' (15dB), 'medium' (12dB), 'high' (9dB)
        """
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        
        # Sensitivity presets
        sensitivity_map = {'low': 15, 'medium': 12, 'high': 9}
        self.threshold_db = sensitivity_map.get(sensitivity, threshold_db)
        
        self.alarm_sound = alarm_sound
        self.running = False
        self.detection_active = False
        
        # Statistics
        self.detection_history = deque(maxlen=100)
        self.last_detection_time = 0
        self.total_detections = 0
        
        # Initialize pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=1)
        self._load_alarm_sound()
        
        # Initialize SDR
        self.sdr = None
        
    def _load_alarm_sound(self):
        """Laad alarm geluid of genereer beep"""
        if self.alarm_sound:
            try:
                self.alarm = pygame.mixer.Sound(self.alarm_sound)
            except:
                print(f"Kan {self.alarm_sound} niet laden, gebruik beep")
                self._generate_beep()
        else:
            self._generate_beep()
    
    def _generate_beep(self):
        """Genereer alarm beep (1000 Hz, 500ms)"""
        sample_rate = 22050
        duration = 0.5
        frequency = 1000
        
        samples = np.sin(2 * np.pi * frequency * 
                        np.linspace(0, duration, int(sample_rate * duration)))
        
        # Envelope voor minder harsh geluid
        envelope = np.linspace(1, 0, len(samples))
        samples = (samples * envelope * 32767).astype(np.int16)
        
        # Stereo (dupliceer voor beide kanalen)
        stereo_samples = np.column_stack((samples, samples))
        
        self.alarm = pygame.sndarray.make_sound(stereo_samples)
    
    def init_sdr(self, gain='auto', ppm_error=0):
        """Initialiseer RTL-SDR"""
        try:
            self.sdr = RtlSdr()
            self.sdr.sample_rate = self.sample_rate
            self.sdr.center_freq = self.center_freq
            
            if gain == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(gain)
            
            self.sdr.freq_correction = ppm_error
            
            print(f"✓ RTL-SDR geïnitialiseerd")
            print(f"  Center freq: {self.center_freq/1e6:.2f} MHz")
            print(f"  Sample rate: {self.sample_rate/1e6:.2f} MS/s")
            print(f"  Gain: {self.sdr.gain}")
            
            # Warm-up samples (eerst samples zijn vaak corrupt)
            self.sdr.read_samples(4096)
            
            return True
        except Exception as e:
            print(f"✗ Fout bij initialiseren SDR: {e}")
            return False
    
    def detect_signals(self, samples, visualize=False):
        """
        Detecteer TETRA signalen in samples
        
        Returns:
            (detected, strength_db, frequencies)
        """
        # Bereken power spectral density
        freqs, psd = welch(samples, fs=self.sample_rate, 
                          nperseg=2048, scaling='density')
        psd_db = 10 * np.log10(psd + 1e-20)  # Voorkom log(0)
        
        # Schat ruisvloer (25e percentiel = robuust tegen signalen)
        noise_floor = np.percentile(psd_db, 25)
        
        # Detecteer pieken boven threshold
        threshold = noise_floor + self.threshold_db
        peaks, properties = find_peaks(psd_db, height=threshold, 
                                       prominence=3, distance=10)
        
        if len(peaks) > 0:
            # Bereken absolute frequenties
            detected_freqs = freqs[peaks] + self.center_freq
            peak_strengths = psd_db[peaks]
            max_strength = np.max(peak_strengths) - noise_floor
            
            if visualize:
                self._print_spectrum(freqs, psd_db, peaks, noise_floor, threshold)
            
            return True, max_strength, detected_freqs
        
        return False, 0, []
    
    def _print_spectrum(self, freqs, psd_db, peaks, noise_floor, threshold):
        """Print ASCII spectrum analyzer"""
        print("\n" + "="*60)
        print("SPECTRUM ANALYZER")
        print("="*60)
        
        # Selecteer 50 punten voor display
        indices = np.linspace(0, len(psd_db)-1, 50, dtype=int)
        display_psd = psd_db[indices]
        
        # Normaliseer voor display (0-20 tekens hoog)
        normalized = ((display_psd - np.min(display_psd)) / 
                     (np.max(display_psd) - np.min(display_psd)) * 20)
        
        for level in range(20, -1, -1):
            line = ""
            for val in normalized:
                if val >= level:
                    line += "█"
                elif val >= level - 0.5:
                    line += "▄"
                else:
                    line += " "
            
            # Labels
            if level == 20:
                line += f" {np.max(display_psd):.1f} dB"
            elif level == 10:
                line += f" {threshold:.1f} dB (threshold)"
            elif level == 0:
                line += f" {np.min(display_psd):.1f} dB"
            
            print(line)
        
        print("-" * 50)
        print(f"Noise floor: {noise_floor:.1f} dB | Detecties: {len(peaks)}")
        
        if len(peaks) > 0:
            print("\nGedetecteerde frequenties:")
            for peak_idx in peaks[:5]:  # Max 5 tonen
                freq = (freqs[peak_idx] + self.center_freq) / 1e6
                strength = psd_db[peak_idx]
                print(f"  {freq:.3f} MHz @ {strength:.1f} dB")
    
    def trigger_alarm(self, strength_db, frequencies):
        """Trigger alarm en log detectie"""
        current_time = time.time()
        
        # Debounce: max 1 alarm per 3 seconden
        if current_time - self.last_detection_time < 3:
            return
        
        self.last_detection_time = current_time
        self.total_detections += 1
        
        # Log detectie
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n🚨 [{timestamp}]  DETECTED!")
        print(f"   Signaalsterkte: {strength_db:.1f} dB boven ruisvloer")
        print(f"   Frequenties: {', '.join([f'{f/1e6:.3f} MHz' for f in frequencies[:3]])}")
        print(f"   Totaal detecties: {self.total_detections}\n")
        
        # Speel alarm
        if not self.detection_active:
            self.detection_active = True
            threading.Thread(target=self._play_alarm_sequence).start()
    
    def _play_alarm_sequence(self):
        """Speel alarm 3x achter elkaar"""
        for i in range(3):
            self.alarm.play()
            time.sleep(0.6)
        self.detection_active = False
    
    def monitor_loop(self, duration=None, visualize_interval=10):
        """
        Hoofdloop: monitor TETRA band continu
        
        Args:
            duration: Hoelang monitoren in seconden (None = oneindig)
            visualize_interval: Toon spectrum elke N detecties
        """
        if not self.sdr:
            print("✗ SDR niet geïnitialiseerd! Roep eerst init_sdr() aan")
            return
        
        self.running = True
        start_time = time.time()
        loop_count = 0
        
        print("\n" + "="*60)
        print("TETRA MONITOR ACTIEF - Druk Ctrl+C om te stoppen")
        print("="*60)
        print(f"Threshold: {self.threshold_db} dB boven ruisvloer")
        print(f"Center freq: {self.center_freq/1e6:.2f} MHz")
        print("Monitoring...")
        
        try:
            while self.running:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break
                
                # Verzamel samples
                samples = self.sdr.read_samples(128*1024)
                
                # Detecteer signalen
                visualize = (loop_count % visualize_interval == 0)
                detected, strength, frequencies = self.detect_signals(samples, visualize)
                
                if detected:
                    self.trigger_alarm(strength, frequencies)
                    self.detection_history.append(time.time())
                
                loop_count += 1
                
                # Korte pauze om CPU te sparen
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n\nGestopt door gebruiker")
        finally:
            self.stop()
    
    def stop(self):
        """Stop monitoring en sluit SDR"""
        self.running = False
        if self.sdr:
            self.sdr.close()
            print("✓ SDR afgesloten")
        
        # Print statistieken
        print("\n" + "="*60)
        print("SESSIE STATISTIEKEN")
        print("="*60)
        print(f"Totaal detecties: {self.total_detections}")
        
        if len(self.detection_history) > 1:
            intervals = np.diff(list(self.detection_history))
            print(f"Gemiddeld interval: {np.mean(intervals):.1f} seconden")
            print(f"Kortste interval: {np.min(intervals):.1f} seconden")


def main():
    parser = argparse.ArgumentParser(
        description='TETRA/C2000 Detector voor 112 Spotting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Voorbeelden:
  # Standaard monitoring (medium sensitivity)
  python3 tetra_alarm.py
  
  # Hoge gevoeligheid (detecteert verder weg)
  python3 tetra_alarm.py --sensitivity high
  
  # Custom threshold en frequentie
  python3 tetra_alarm.py --threshold 10 --freq 392.5
  
  # Met custom alarm geluid
  python3 tetra_alarm.py --alarm siren.wav
  
  # Test mode (1 minuut, spectrum visualisatie)
  python3 tetra_alarm.py --duration 60 --viz-interval 1
        """)
    
    parser.add_argument('--freq', type=float, default=390.0,
                       help='Center frequentie in MHz (default: 390)')
    parser.add_argument('--threshold', type=float,
                       help='Threshold in dB boven ruisvloer')
    parser.add_argument('--sensitivity', choices=['low', 'medium', 'high'],
                       default='medium',
                       help='Sensitivity preset (default: medium)')
    parser.add_argument('--gain', default='auto',
                       help='RTL-SDR gain (auto of 0-49.6)')
    parser.add_argument('--ppm', type=int, default=0,
                       help='PPM frequency correction')
    parser.add_argument('--alarm', type=str,
                       help='Pad naar custom alarm .wav bestand')
    parser.add_argument('--duration', type=int,
                       help='Monitor duration in seconden (default: oneindig)')
    parser.add_argument('--viz-interval', type=int, default=10,
                       help='Toon spectrum elke N detecties')
    
    args = parser.parse_args()
    
    # Maak detector
    kwargs = {
        'center_freq': args.freq * 1e6,
        'sensitivity': args.sensitivity,
    }
    
    if args.threshold:
        kwargs['threshold_db'] = args.threshold
    if args.alarm:
        kwargs['alarm_sound'] = args.alarm
    
    detector = TetraDetector(**kwargs)
    
    # Initialiseer SDR
    if not detector.init_sdr(gain=args.gain, ppm_error=args.ppm):
        return 1
    
    # Start monitoring
    detector.monitor_loop(duration=args.duration, 
                         visualize_interval=args.viz_interval)
    
    return 0


if __name__ == '__main__':
    exit(main())
      
